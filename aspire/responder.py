# aspire.responders | Csware

import asyncio
import functools
import concurrent.futures
import multiprocessing
import traceback

import jinja2

import json
import re
import inspect
import typing


#import functools
import io

import gzip

import chardet
import rfc3986
import graphene
import yaml

from contextlib import contextmanager
from urllib.parse import parse_qs, urlencode
from base64 import b64decode
from http.cookies import SimpleCookie
from requests_toolbelt.multipart import decoder


from aspire.core.reactor import (
     Response as Responder,
    StreamingResponse as ResponseStream,
    Request as Requester, 
    State,
    run_in_threadpool, HTTPException, Lifespan, State, StaticFiles, 
    WSGIMiddleware, WebSocket, WebSocketClose
)
from aspire import status_codes
from aspire.config import DEFAULT_SESSION_COOKIE, DEFAULT_API_THEME

from requests.structures import CaseInsensitiveDict
from requests.cookies import RequestsCookieJar



from aspire.config import DEFAULT_ENCODING
from aspire.status_codes import HTTP_200, HTTP_301



#----------------------------- models -------------------------------------
class QueryDict(dict):
    def __init__(self, query_string):
        self.update(parse_qs(query_string))

    def __getitem__(self, key):
        """
        Return the last data value for this key, or [] if it's an empty list;
        raise KeyError if not found.
        """
        list_ = super().__getitem__(key)
        try:
            return list_[-1]
        except IndexError:
            return []

    def get(self, key, default=None):
        """
        Return the last data value for the passed key. If key doesn't exist
        or value is an empty list, return `default`.
        """
        try:
            val = self[key]
        except KeyError:
            return default
        if val == []:
            return default
        return val

    def _get_list(self, key, default=None, force_list=False):
        """
        Return a list of values for the key.

        Used internally to manipulate values list. If force_list is True,
        return a new copy of values.
        """
        try:
            values = super().__getitem__(key)
        except KeyError:
            if default is None:
                return []
            return default
        else:
            if force_list:
                values = list(values) if values is not None else None
            return values

    def get_list(self, key, default=None):
        """
        Return the list of values for the key. If key doesn't exist, return a
        default value.
        """
        return self._get_list(key, default, force_list=True)

    def items(self):
        """
        Yield (key, value) pairs, where value is the last item in the list
        associated with the key.
        """
        for key in self:
            yield key, self[key]

    def items_list(self):
        """
        Yield (key, value) pairs, where value is the the list.
        """
        yield from super().items()


class Request:
    __slots__ = [
        "_aspire",
        "formats",
        "_headers",
        "_encoding",
        "api",
        "_content",
        "_cookies",
    ]

    def __init__(self, scope, receive, api=None, formats=None):
        self._aspire = Requester(scope, receive)
        self.formats = formats
        self._encoding = None
        self.api = api
        self._content = None

        headers = CaseInsensitiveDict()
        for key, value in self._aspire.headers.items():
            headers[key] = value

        self._headers = headers
        self._cookies = None

    @property
    def session(self):
        """The session data, in dict form, from the Request."""
        return self._aspire.session

    @property
    def headers(self):
        """A case-insensitive dictionary, containing all headers sent in the Request."""
        return self._headers

    @property
    def mimetype(self):
        return self.headers.get("Content-Type", "")

    @property
    def method(self):
        """The incoming HTTP method used for the request, lower-cased."""
        return self._aspire.method.lower()

    @property
    def full_url(self):
        """The full URL of the Request, query parameters and all."""
        return str(self._aspire.url)

    @property
    def url(self):
        """The parsed URL of the Request."""
        return rfc3986.urlparse(self.full_url)

    @property
    def cookies(self):
        """The cookies sent in the Request, as a dictionary."""
        if self._cookies is None:
            cookies = RequestsCookieJar()
            cookie_header = self.headers.get("Cookie", "")

            bc = SimpleCookie(cookie_header)
            for key, morsel in bc.items():
                cookies[key] = morsel.value

            self._cookies = cookies.get_dict()

        return self._cookies

    @property
    def params(self):
        """A dictionary of the parsed query parameters used for the Request."""
        try:
            return QueryDict(self.url.query)
        except AttributeError:
            return QueryDict({})

    @property
    def state(self) -> State:
        """
        Use the state to store additional information.

        This can be a very helpful feature, if you want to hand over
        information from a middelware or a route decorator to the
        actual route handler.

        Usage: ``request.state.time_started = time.time()``
        """
        return self._aspire.state

    @property
    async def encoding(self):
        """The encoding of the Request's body. Can be set, manually. Must be awaited."""
        # Use the user-set encoding first.
        if self._encoding:
            return self._encoding

        return await self.apparent_encoding

    @encoding.setter
    def encoding(self, value):
        self._encoding = value

    @property
    async def content(self):
        """The Request body, as bytes. Must be awaited."""
        if not self._content:
            self._content = await self._aspire.body()
        return self._content

    @property
    async def text(self):
        """The Request body, as unicode. Must be awaited."""
        return (await self.content).decode(await self.encoding)

    @property
    async def declared_encoding(self):
        if "Encoding" in self.headers:
            return self.headers["Encoding"]

    @property
    async def apparent_encoding(self):
        """The apparent encoding, provided by the chardet library. Must be awaited."""
        declared_encoding = await self.declared_encoding

        if declared_encoding:
            return declared_encoding

        return chardet.detect(await self.content)["encoding"] or DEFAULT_ENCODING

    @property
    def is_secure(self):
        return self.url.scheme == "https"

    def accepts(self, content_type):
        """Returns ``True`` if the incoming Request accepts the given ``content_type``."""
        return content_type in self.headers.get("Accept", [])

    async def media(self, format=None):
        """Renders incoming json/yaml/form data as Python objects. Must be awaited.

        :param format: The name of the format being used. Alternatively accepts a custom callable for the format type.
        """

        if format is None:
            format = "yaml" if "yaml" in self.mimetype or "" else "json"
            format = "form" if "form" in self.mimetype or "" else format

        if format in self.formats:
            return await self.formats[format](self)
        else:
            return await format(self)


def content_setter(mimetype):
    def getter(instance):
        return instance.content

    def setter(instance, value):
        instance.content = value
        instance.mimetype = mimetype

    return property(fget=getter, fset=setter)


class Response:
    __slots__ = [
        "req",
        "status_code",
        "content",
        "encoding",
        "media",
        "headers",
        "formats",
        "cookies",
        "session",
        "mimetype",
        "_stream",
    ]

    text = content_setter("text/plain")
    html = content_setter("text/html")

    def __init__(self, req, *, formats):
        self.req = req
        self.status_code = None  #: The HTTP Status Code to use for the Response.
        self.content = None  #: A bytes representation of the response body.
        self.mimetype = None
        self.encoding = DEFAULT_ENCODING
        self.media = None  #: A Python object that will be content-negotiated and sent back to the client. Typically, in JSON formatting.
        self._stream = None
        self.headers = (
            {}
        )  #: A Python dictionary of ``{key: value}``, representing the headers of the response.
        self.formats = formats
        self.cookies = SimpleCookie()  #: The cookies set in the Response
        self.session = (
            req.session
        )  #: The cookie-based session data, in dict form, to add to the Response.

    # Property or func/dec
    def stream(self, func, *args, **kwargs):
        assert inspect.isasyncgenfunction(func)

        self._stream = functools.partial(func, *args, **kwargs)

        return func

    def redirect(self, location, *, set_text=True, status_code=HTTP_301):
        self.status_code = status_code
        if set_text:
            self.text = f"Redirecting to: {location}"
        self.headers.update({"Location": location})

    @property
    async def body(self):
        if self._stream is not None:
            return (self._stream(), {})

        if self.content is not None:
            headers = {}
            content = self.content
            if self.mimetype is not None:
                headers["Content-Type"] = self.mimetype
            if self.mimetype == "text/plain" and self.encoding is not None:
                headers["Encoding"] = self.encoding
                content = content.encode(self.encoding)
            return (content, headers)

        for format in self.formats:
            if self.req.accepts(format):
                return (await self.formats[format](self, encode=True)), {}

        # Default to JSON.
        return (
            await self.formats["json"](self, encode=True),
            {"Content-Type": "application/json"},
        )

    def set_cookie(
        self,
        key,
        value="",
        expires=None,
        path="/",
        domain=None,
        max_age=None,
        secure=True,
        httponly=True,
    ):
        self.cookies[key] = value
        morsel = self.cookies[key]
        if expires is not None:
            morsel["expires"] = expires
        if path is not None:
            morsel["path"] = path
        if domain is not None:
            morsel["domain"] = domain
        if max_age is not None:
            morsel["max-age"] = max_age
        morsel["secure"] = secure
        morsel["httponly"] = httponly

    def _prepare_cookies(self, aspire_response):
        cookie_header = (
            (b"set-cookie", morsel.output(header="").lstrip().encode("latin-1"))
            for morsel in self.cookies.values()
        )
        aspire_response.raw_headers.extend(cookie_header)

    async def __call__(self, scope, receive, send):
        body, headers = await self.body
        if self.headers:
            headers.update(self.headers)

        if self._stream is not None:
            response_cls =ResponseStream
        else:
            response_cls = Responder

        response = response_cls(body, status_code=self.status_code, headers=headers)
        self._prepare_cookies(response)

        await response(scope, receive, send)


#----------------------- Formats ---------------------------------------------------
async def format_form(r, encode=False):
    if encode:
        pass
    elif "multipart/form-data" in r.headers.get("Content-Type"):
        decode = decoder.MultipartDecoder(await r.content, r.mimetype)
        querys = list()
        for part in decode.parts:
            header = part.headers.get(b"Content-Disposition").decode("utf-8")
            text = part.text

            for section in [h.strip() for h in header.split(";")]:
                split = section.split("=")
                if len(split) > 1:
                    key = split[1]
                    key = key[1:-1]
                    querys.append((key, text))

        content = urlencode(querys)
        return QueryDict(content)
    else:
        return QueryDict(await r.text)


async def format_yaml(r, encode=False):
    if encode:
        r.headers.update({"Content-Type": "application/x-yaml"})
        return yaml.safe_dump(r.media)
    else:
        return yaml.safe_load(await r.content)


async def format_json(r, encode=False):
    if encode:
        r.headers.update({"Content-Type": "application/json"})
        return json.dumps(r.media)
    else:
        return json.loads(await r.content)


async def format_files(r, encode=False):
    if encode:
        pass
    else:
        decoded = decoder.MultipartDecoder(await r.content, r.mimetype)
        dump = {}
        for part in decoded.parts:
            header = part.headers[b"Content-Disposition"].decode("utf-8")
            mimetype = part.headers.get(b"Content-Type", None)
            filename = None

            for section in [h.strip() for h in header.split(";")]:
                split = section.split("=")
                if len(split) > 1:
                    key = split[0]
                    value = split[1]

                    value = value[1:-1]

                    if key == "filename":
                        filename = value
                    elif key == "name":
                        formname = value

            if mimetype is None:
                dump[formname] = part.content
            else:
                dump[formname] = {
                    "filename": filename,
                    "content": part.content,
                    "content-type": mimetype.decode("utf-8"),
                }
        return dump


def get_formats():
    return {
        "json": format_json,
        "yaml": format_yaml,
        "form": format_form,
        "files": format_files,
    }


#----------------------- Formats ---------------------------------------------------
async def format_form(r, encode=False):
    if encode:
        pass
    elif "multipart/form-data" in r.headers.get("Content-Type"):
        decode = decoder.MultipartDecoder(await r.content, r.mimetype)
        querys = list()
        for part in decode.parts:
            header = part.headers.get(b"Content-Disposition").decode("utf-8")
            text = part.text

            for section in [h.strip() for h in header.split(";")]:
                split = section.split("=")
                if len(split) > 1:
                    key = split[1]
                    key = key[1:-1]
                    querys.append((key, text))

        content = urlencode(querys)
        return QueryDict(content)
    else:
        return QueryDict(await r.text)


async def format_yaml(r, encode=False):
    if encode:
        r.headers.update({"Content-Type": "application/x-yaml"})
        return yaml.safe_dump(r.media)
    else:
        return yaml.safe_load(await r.content)


async def format_json(r, encode=False):
    if encode:
        r.headers.update({"Content-Type": "application/json"})
        return json.dumps(r.media)
    else:
        return json.loads(await r.content)


async def format_files(r, encode=False):
    if encode:
        pass
    else:
        decoded = decoder.MultipartDecoder(await r.content, r.mimetype)
        dump = {}
        for part in decoded.parts:
            header = part.headers[b"Content-Disposition"].decode("utf-8")
            mimetype = part.headers.get(b"Content-Type", None)
            filename = None

            for section in [h.strip() for h in header.split(";")]:
                split = section.split("=")
                if len(split) > 1:
                    key = split[0]
                    value = split[1]

                    value = value[1:-1]

                    if key == "filename":
                        filename = value
                    elif key == "name":
                        formname = value

            if mimetype is None:
                dump[formname] = part.content
            else:
                dump[formname] = {
                    "filename": filename,
                    "content": part.content,
                    "content-type": mimetype.decode("utf-8"),
                }
        return dump


def get_formats():
    return {
        "json": format_json,
        "yaml": format_yaml,
        "form": format_form,
        "files": format_files,
    }


#----------------------------- Background Response --------------------
class BackgroundQueue:
    def __init__(self, n=None):
        if n is None:
            n = multiprocessing.cpu_count()

        self.n = n
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=n)
        self.results = []

    def run(self, f, *args, **kwargs):
        self.pool._max_workers = self.n
        self.pool._adjust_thread_count()

        f = self.pool.submit(f, *args, **kwargs)
        self.results.append(f)
        return f

    def task(self, f):
        def on_future_done(fs):
            try:
                fs.result()
            except:
                traceback.print_exc()

        def do_task(*args, **kwargs):
            result = self.run(f, *args, **kwargs)
            result.add_done_callback(on_future_done)
            return result

        return do_task

    async def __call__(self, func, *args, **kwargs) -> None:
        if asyncio.iscoroutinefunction(func):
            return await asyncio.ensure_future(func(*args, **kwargs))
        else:
            return await run_in_threadpool(func, *args, **kwargs)



#---------------------------- Router Configurations -------------------------------

_CONVERTORS = {
    "int": (int, r"\d+"),
    "str": (str, r"[^/]+"),
    "float": (float, r"\d+(.\d+)?"),
}

PARAM_RE = re.compile("{([a-zA-Z_][a-zA-Z0-9_]*)(:[a-zA-Z_][a-zA-Z0-9_]*)?}")



#--------------------------------- Routes -----------------------------------
def compile_path(path):
    path_re = "^"
    param_convertors = {}
    idx = 0

    for match in PARAM_RE.finditer(path):
        param_name, convertor_type = match.groups(default="str")
        convertor_type = convertor_type.lstrip(":")
        assert (
            convertor_type in _CONVERTORS.keys()
        ), f"Unknown path convertor '{convertor_type}'"
        convertor, convertor_re = _CONVERTORS[convertor_type]

        path_re += path[idx : match.start()]
        path_re += rf"(?P<{param_name}>{convertor_re})"

        param_convertors[param_name] = convertor

        idx = match.end()

    path_re += path[idx:] + "$"

    return re.compile(path_re), param_convertors


class BaseRoute:
    def matches(self, scope):
        raise NotImplementedError()

    async def __call__(self, scope, receive, send):
        raise NotImplementedError()


class Route(BaseRoute):
    def __init__(self, route, endpoint, *, before_request=False):
        assert route.startswith("/"), "Route path must start with '/'"
        self.route = route
        self.endpoint = endpoint
        self.before_request = before_request

        self.path_re, self.param_convertors = compile_path(route)

    def __repr__(self):
        return f"<Route {self.route!r}={self.endpoint!r}>"

    def url(self, **params):
        return self.route.format(**params)

    @property
    def endpoint_name(self):
        return self.endpoint.__name__

    @property
    def description(self):
        return self.endpoint.__doc__

    def matches(self, scope):
        if scope["type"] != "http":
            return False, {}

        path = scope["path"]
        match = self.path_re.match(path)

        if match is None:
            return False, {}

        matched_params = match.groupdict()
        for key, value in matched_params.items():
            matched_params[key] = self.param_convertors[key](value)

        return True, {"path_params": {**matched_params}}

    async def __call__(self, scope, receive, send):
        request = Request(scope, receive, formats=get_formats())
        response = Response(req=request, formats=get_formats())

        path_params = scope.get("path_params", {})
        before_requests = scope.get("before_requests", [])

        for before_request in before_requests.get("http", []):
            if asyncio.iscoroutinefunction(before_request):
                await before_request(request, response)
            else:
                await run_in_threadpool(before_request, request, response)

        views = []

        if inspect.isclass(self.endpoint):
            endpoint = self.endpoint()
            on_request = getattr(endpoint, "on_request", None)
            if on_request:
                views.append(on_request)

            method_name = f"on_{request.method}"
            try:
                view = getattr(endpoint, method_name)
                views.append(view)
            except AttributeError:
                if on_request is None:
                    raise HTTPException(status_code=status_codes.HTTP_405)
        else:
            views.append(self.endpoint)

        for view in views:
            # "Monckey patch" for graphql: explicitly checking __call__
            if asyncio.iscoroutinefunction(view) or asyncio.iscoroutinefunction(
                view.__call__
            ):
                await view(request, response, **path_params)
            else:
                await run_in_threadpool(view, request, response, **path_params)

        if response.status_code is None:
            response.status_code = status_codes.HTTP_200

        await response(scope, receive, send)

    def __eq__(self, other):
        # [TODO] compare to str ?
        return self.route == other.route and self.endpoint == other.endpoint

    def __hash__(self):
        return hash(self.route) ^ hash(self.endpoint) ^ hash(self.before_request)


class WebSocketRoute(BaseRoute):
    def __init__(self, route, endpoint, *, before_request=False):
        assert route.startswith("/"), "Route path must start with '/'"
        self.route = route
        self.endpoint = endpoint
        self.before_request = before_request

        self.path_re, self.param_convertors = compile_path(route)

    def __repr__(self):
        return f"<Route {self.route!r}={self.endpoint!r}>"

    def url(self, **params):
        return self.route.format(**params)

    @property
    def endpoint_name(self):
        return self.endpoint.__name__

    @property
    def description(self):
        return self.endpoint.__doc__

    def matches(self, scope):
        if scope["type"] != "websocket":
            return False, {}

        path = scope["path"]
        match = self.path_re.match(path)

        if match is None:
            return False, {}

        matched_params = match.groupdict()
        for key, value in matched_params.items():
            matched_params[key] = self.param_convertors[key](value)

        return True, {"path_params": {**matched_params}}

    async def __call__(self, scope, receive, send):
        ws = WebSocket(scope, receive, send)

        before_requests = scope.get("before_requests", [])
        for before_request in before_requests.get("ws", []):
            await before_request(ws)

        await self.endpoint(ws)

    def __eq__(self, other):
        # [TODO] compare to str ?
        return self.route == other.route and self.endpoint == other.endpoint

    def __hash__(self):
        return hash(self.route) ^ hash(self.endpoint) ^ hash(self.before_request)


class Router:
    def __init__(self, routes=None, default_response=None, before_requests=None):
        self.routes = [] if routes is None else list(routes)
        # [TODO] Make its own router
        self.apps = {}
        self.default_endpoint = (
            self.default_response if default_response is None else default_response
        )
        self.lifespan_handler = Lifespan()
        self.before_requests = (
            {"http": [], "ws": []} if before_requests is None else before_requests
        )

    def add_route(
        self,
        route=None,
        endpoint=None,
        *,
        default=False,
        websocket=False,
        before_request=False,
        check_existing=False,
    ):
        """ Adds a route to the router.
        :param route: A string representation of the route
        :param endpoint: The endpoint for the route -- can be callable, or class.
        :param default: If ``True``, all unknown requests will route to this view.
        """
        if before_request:
            if websocket:
                self.before_requests.setdefault("ws", []).append(endpoint)
            else:
                self.before_requests.setdefault("http", []).append(endpoint)
            return

        if check_existing:
            assert not self.routes or route not in (
                item.route for item in self.routes
            ), f"Route '{route}' already exists"

        if default:
            self.default_endpoint = endpoint

        if websocket:
            route = WebSocketRoute(route, endpoint)
        else:
            route = Route(route, endpoint)

        self.routes.append(route)

    def mount(self, route, app):
        """Mounts ASGI / WSGI applications at a given route
        """
        self.apps.update(route, app)

    def before_request(self, endpoint, websocket=False):
        if websocket:
            self.before_requests.setdefault("ws", []).append(endpoint)
        else:
            self.before_requests.setdefault("http", []).append(endpoint)

    def url_for(self, endpoint, **params):
        # TODO: Check for params
        for route in self.routes:
            if endpoint in (route.endpoint, route.endpoint.__name__):
                return route.url(**params)
        return None

    async def default_response(self, scope, receive, send):
        if scope["type"] == "websocket":
            websocket_close = WebSocketClose()
            await websocket_close(receive, send)
            return

        request = Request(scope, receive)
        response = Response(request, formats=get_formats())

        raise HTTPException(status_code=status_codes.HTTP_404)

    def _resolve_route(self, scope):
        for route in self.routes:
            matches, child_scope = route.matches(scope)
            if matches:
                scope.update(child_scope)
                return route
        return None

    async def __call__(self, scope, receive, send):
        assert scope["type"] in ("http", "websocket", "lifespan")

        if scope["type"] == "lifespan":
            await self.lifespan_handler(scope, receive, send)
            return

        path = scope["path"]
        root_path = scope.get("root_path", "")

        # Check "primary" mounted routes first (before submounted apps)
        route = self._resolve_route(scope)

        scope["before_requests"] = self.before_requests

        if route is not None:
            await route(scope, receive, send)
            return

        # Call into a submounted app, if one exists.
        for path_prefix, app in self.apps.items():
            if path.startswith(path_prefix):
                scope["path"] = path[len(path_prefix) :]
                scope["root_path"] = root_path + path_prefix
                try:
                    await app(scope, receive, send)
                    return
                except TypeError:
                    app = WSGIMiddleware(app)
                    await app(scope, receive, send)
                    return

        await self.default_response(scope, receive, send)


#------------------------------ Static Files ------------------------------------
class StaticFiles(StaticFiles):
    """I've created an issue to disccuss allowing multiple directories in starletter's `StaticFiles`.
    
    https://github.com/encode/starlette/issues/625
    
    I've also made a PR to add this method to starlette StaticFiles
    Once accepted we will remove this.
    
    https://github.com/encode/starlette/pull/626
    """

    def add_directory(self, directory: str) -> None:
        self.all_directories = [*self.all_directories, *self.get_directories(directory)]



#-------------------------------- Templates -------------------------------------------------------
class Templates:
    def __init__(
        self, directory="templates", autoescape=True, context=None, enable_async=False
    ):
        self.directory = directory
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader([str(self.directory)]),
            autoescape=autoescape,
            enable_async=enable_async,
        )
        self.default_context = {} if context is None else {**context}
        self._env.globals.update(self.default_context)

    @property
    def context(self):
        return self._env.globals

    @context.setter
    def context(self, context):
        self._env.globals = {**self.default_context, **context}

    def get_template(self, name):
        return self._env.get_template(name)

    def render(self, template, *args, **kwargs):
        """Renders the given `jinja2 <http://jinja.pocoo.org/docs/>`_ template, with provided values supplied.

        :param template: The filename of the jinja2 template.
        :param **kwargs: Data to pass into the template.
        :param **kwargs: Data to pass into the template.
        """
        return self.get_template(template).render(*args, **kwargs)

    @contextmanager
    def _async(self):
        self._env.is_async = True
        try:
            yield
        finally:
            self._env.is_async = False

    async def render_async(self, template, *args, **kwargs):
        with self._async():
            return await self.get_template(template).render_async(*args, **kwargs)

    def render_string(self, source, *args, **kwargs):
        """Renders the given `jinja2 <http://jinja.pocoo.org/docs/>`_ template string, with provided values supplied.

        :param source: The template to use.
        :param *args, **kwargs: Data to pass into the template.
        :param **kwargs: Data to pass into the template.
        """
        template = self._env.from_string(source)
        return template.render(*args, **kwargs)
