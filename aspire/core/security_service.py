# aspire.core.security.py | Aspire Security Services | Csware | Feb 5 2020 | MIT

import asyncio
import functools
import inspect

import itsdangerous
import json
import re
import typing

from base64 import b64decode, b64encode
from cryptography.fernet import Fernet as fn
from itsdangerous.exc import BadTimeSignature, SignatureExpired
from strgen import StringGenerator


from aspire.core.reactor import (
    Headers,     
    HTTPConnection,  
    HTTPException,  
    MutableHeaders,   
    PlainTextResponse,
    RedirectResponse, 
    Request,   
    Response, 
    Secret,
    URL,
    WebSocket,
    ASGIApp, 
    Message,
    Receive, 
    Scope, 
    Send

)


#------------------------------ Utilities -------------------------------------
def has_required_scope(conn: HTTPConnection, scopes: typing.Sequence[str]) -> bool:
    for scope in scopes:
        if scope not in conn.auth.scopes:
            return False
    return True


def requires(
    scopes: typing.Union[str, typing.Sequence[str]],
    status_code: int = 403,
    redirect: str = None,
) -> typing.Callable:
    scopes_list = [scopes] if isinstance(scopes, str) else list(scopes)

    def decorator(func: typing.Callable) -> typing.Callable:
        type = None
        sig = inspect.signature(func)
        for idx, parameter in enumerate(sig.parameters.values()):
            if parameter.name == "request" or parameter.name == "websocket":
                type = parameter.name
                break
        else:
            raise Exception(
                f'No "request" or "websocket" argument on function "{func}"'
            )

        if type == "websocket":
            # Handle websocket functions. (Always async)
            @functools.wraps(func)
            async def websocket_wrapper(
                *args: typing.Any, **kwargs: typing.Any
            ) -> None:
                websocket = kwargs.get("websocket", args[idx])
                assert isinstance(websocket, WebSocket)

                if not has_required_scope(websocket, scopes_list):
                    await websocket.close()
                else:
                    await func(*args, **kwargs)

            return websocket_wrapper

        elif asyncio.iscoroutinefunction(func):
            # Handle async request/response functions.
            @functools.wraps(func)
            async def async_wrapper(
                *args: typing.Any, **kwargs: typing.Any
            ) -> Response:
                request = kwargs.get("request", args[idx])
                assert isinstance(request, Request)

                if not has_required_scope(request, scopes_list):
                    if redirect is not None:
                        return RedirectResponse(url=request.url_for(redirect))
                    raise HTTPException(status_code=status_code)
                return await func(*args, **kwargs)

            return async_wrapper

        else:
            # Handle sync request/response functions.
            @functools.wraps(func)
            def sync_wrapper(*args: typing.Any, **kwargs: typing.Any) -> Response:
                request = kwargs.get("request", args[idx])
                assert isinstance(request, Request)

                if not has_required_scope(request, scopes_list):
                    if redirect is not None:
                        return RedirectResponse(url=request.url_for(redirect))
                    raise HTTPException(status_code=status_code)
                return func(*args, **kwargs)

            return sync_wrapper

    return decorator


#------------------------------ Authentication Services ------------------------
class AuthenticationError(Exception):
    pass



class AuthenticationService:
    """Authentication 

        User Authentication Implementation
        Auth
    """
    def __init__(self):
        super(AuthenticationService, self).__init__()
    
    async def authenticate(
        self, conn: HTTPConnection
    ) -> typing.Optional[typing.Tuple["AuthCredentials", "BaseUser"]]:
        raise NotImplementedError()  # pragma: no cover



class AuthenticationHandler:
    """Authentication 

        User Authentication Implementation
    """
    def __init__(self):
        super(AuthenticationHandler, self).__init__()


class AuthCredentials:
    def __init__(self, scopes: typing.Sequence[str] = None):
        self.scopes = [] if scopes is None else list(scopes)


class BaseUser:
    @property
    def is_authenticated(self) -> bool:
        raise NotImplementedError()  # pragma: no cover

    @property
    def display_name(self) -> str:
        raise NotImplementedError()  # pragma: no cover

    @property
    def identity(self) -> str:
        raise NotImplementedError()  # pragma: no cover


class SimpleUser(BaseUser):
    def __init__(self, username: str) -> None:
        self.username = username

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def display_name(self) -> str:
        return self.username


class UnauthenticatedUser(BaseUser):
    @property
    def is_authenticated(self) -> bool:
        return False

    @property
    def display_name(self) -> str:
        return ""



class AuthenticationMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        backend: AuthenticationService,
        on_error: typing.Callable[
            [HTTPConnection, AuthenticationError], Response
        ] = None,
    ) -> None:
        self.app = app
        self.backend = backend
        self.on_error = (
            on_error if on_error is not None else self.default_on_error
        )  # type: typing.Callable[[HTTPConnection, AuthenticationError], Response]

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ["http", "websocket"]:
            await self.app(scope, receive, send)
            return

        conn = HTTPConnection(scope)
        try:
            auth_result = await self.backend.authenticate(conn)
        except AuthenticationError as exc:
            response = self.on_error(conn, exc)
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1000})
            else:
                await response(scope, receive, send)
            return

        if auth_result is None:
            auth_result = AuthCredentials(), UnauthenticatedUser()
        scope["auth"], scope["user"] = auth_result
        await self.app(scope, receive, send)

    @staticmethod
    def default_on_error(conn: HTTPConnection, exc: Exception) -> Response:
        return PlainTextResponse(str(exc), status_code=400)


#------------------------------ Security  Services ------------------------

class Secret:
    """
    Holds a string value that should not be revealed in tracebacks etc.
    You should cast the value to `str` at the point it is required.
    """

    def __init__(self, value: str):
        self._value = value

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        return f"{class_name}('**********')"

    def __str__(self) -> str:
        return self._value
        

class Security:
    """ServerCertificate
    
        Handles Retreival and securing the server certificates and encryption keys
        Implements 

    """

    def vault(self):
        """ Vault

            Secure Storage Fascility 
            store security keys, certificates, document signing keys ,and your privacies
            requires authentication
        """


class ServerSecret:
    
    def get_server_cert(self):
        try:
            import os 
        except ImportError:
            return
        print(os.listdir())
        '''cert_path = os.path('.ssl/aspire.crt')
        
        with open(cert_path, 'r') as f:
            cert = f.read()
        f.close()
        del(cert_path)
        del(path)
        del(f)'''
        return #cert 

    def get_server_key(self):
        try:
            from os import path
        except ImportError:
            return
        key_path = path.abspath('core/.ssl/aspire.pem')
        with open(key_path, 'r') as f:
            key = f.read()
        f.close()
        
        del(key_path)
        del(path)
        del(f)
        return key

    def set_server_cert():
        pass

    def set_server_key():
        pass

    def destroy_server_cert():
        pass

    def destroy_server_key():
        pass

    def issue_new_cert():
        pass



#------------------------------ Internet Security Services ------------------------
class SessionService:  # Reimplementation of the SessionMiddleware
    pass

class SessionMiddleware: # Depricated will be replaced by SessionService
    def __init__(
        self,
        app: ASGIApp,
        secret_key: typing.Union[str, Secret],
        session_cookie: str = "session",
        max_age: int = 14 * 24 * 60 * 60,  # 14 days, in seconds
        same_site: str = "lax",
        https_only: bool = False,
    ) -> None:
        self.app = app
        self.signer = itsdangerous.TimestampSigner(str(secret_key))
        self.session_cookie = session_cookie
        self.max_age = max_age
        self.security_flags = "httponly; samesite=" + same_site
        if https_only:  # Secure flag can be used with HTTPS only
            self.security_flags += "; secure"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):  # pragma: no cover
            await self.app(scope, receive, send)
            return

        connection = HTTPConnection(scope)
        initial_session_was_empty = True

        if self.session_cookie in connection.cookies:
            data = connection.cookies[self.session_cookie].encode("utf-8")
            try:
                data = self.signer.unsign(data, max_age=self.max_age)
                scope["session"] = json.loads(b64decode(data))
                initial_session_was_empty = False
            except (BadTimeSignature, SignatureExpired):
                scope["session"] = {}
        else:
            scope["session"] = {}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                if scope["session"]:
                    # We have session data to persist.
                    data = b64encode(json.dumps(scope["session"]).encode("utf-8"))
                    data = self.signer.sign(data)
                    headers = MutableHeaders(scope=message)
                    header_value = "%s=%s; path=/; Max-Age=%d; %s" % (
                        self.session_cookie,
                        data.decode("utf-8"),
                        self.max_age,
                        self.security_flags,
                    )
                    headers.append("Set-Cookie", header_value)
                elif not initial_session_was_empty:
                    # The session has been cleared.
                    headers = MutableHeaders(scope=message)
                    header_value = "%s=%s; %s" % (
                        self.session_cookie,
                        "null; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT;",
                        self.security_flags,
                    )
                    headers.append("Set-Cookie", header_value)
            await send(message)

        await self.app(scope, receive, send_wrapper)


#------------------------------ Network Security Services ------------------------

# Handle Cross Origin Requests

ALL_METHODS = ("DELETE", "GET", "OPTIONS", "PATCH", "POST", "PUT")
ENFORCE_DOMAIN_WILDCARD = "Domain wildcard patterns must be like '*.example.com'."


class CORSMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        allow_origins: typing.Sequence[str] = (),
        allow_methods: typing.Sequence[str] = ("GET", "POST", "PUT", "DELETE", "HEAD"),
        allow_headers: typing.Sequence[str] = (),
        allow_credentials: bool = False,
        allow_origin_regex: str = None,
        expose_headers: typing.Sequence[str] = (),
        max_age: int = 600,
    ) -> None:

        if "*" in allow_methods:
            allow_methods = ALL_METHODS

        compiled_allow_origin_regex = None
        if allow_origin_regex is not None:
            compiled_allow_origin_regex = re.compile(allow_origin_regex)

        simple_headers = {}
        if "*" in allow_origins:
            simple_headers["Access-Control-Allow-Origin"] = "*"
        if allow_credentials:
            simple_headers["Access-Control-Allow-Credentials"] = "true"
        if expose_headers:
            simple_headers["Access-Control-Expose-Headers"] = ", ".join(expose_headers)

        preflight_headers = {}
        if "*" in allow_origins:
            preflight_headers["Access-Control-Allow-Origin"] = "*"
        else:
            preflight_headers["Vary"] = "Origin"
        preflight_headers.update(
            {
                "Access-Control-Allow-Methods": ", ".join(allow_methods),
                "Access-Control-Max-Age": str(max_age),
            }
        )
        if allow_headers and "*" not in allow_headers:
            preflight_headers["Access-Control-Allow-Headers"] = ", ".join(allow_headers)
        if allow_credentials:
            preflight_headers["Access-Control-Allow-Credentials"] = "true"

        self.app = app
        self.allow_origins = allow_origins
        self.allow_methods = allow_methods
        self.allow_headers = [h.lower() for h in allow_headers]
        self.allow_all_origins = "*" in allow_origins
        self.allow_all_headers = "*" in allow_headers
        self.allow_origin_regex = compiled_allow_origin_regex
        self.simple_headers = simple_headers
        self.preflight_headers = preflight_headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # pragma: no cover
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        headers = Headers(scope=scope)
        origin = headers.get("origin")

        if origin is None:
            await self.app(scope, receive, send)
            return

        if method == "OPTIONS" and "access-control-request-method" in headers:
            response = self.preflight_response(request_headers=headers)
            await response(scope, receive, send)
            return

        await self.simple_response(scope, receive, send, request_headers=headers)

    def is_allowed_origin(self, origin: str) -> bool:
        if self.allow_all_origins:
            return True

        if self.allow_origin_regex is not None and self.allow_origin_regex.match(
            origin
        ):
            return True

        return origin in self.allow_origins

    def preflight_response(self, request_headers: Headers) -> Response:
        requested_origin = request_headers["origin"]
        requested_method = request_headers["access-control-request-method"]
        requested_headers = request_headers.get("access-control-request-headers")

        headers = dict(self.preflight_headers)
        failures = []

        if self.is_allowed_origin(origin=requested_origin):
            if not self.allow_all_origins:
                # If self.allow_all_origins is True, then the "Access-Control-Allow-Origin"
                # header is already set to "*".
                # If we only allow specific origins, then we have to mirror back
                # the Origin header in the response.
                headers["Access-Control-Allow-Origin"] = requested_origin
        else:
            failures.append("origin")

        if requested_method not in self.allow_methods:
            failures.append("method")

        # If we allow all headers, then we have to mirror back any requested
        # headers in the response.
        if self.allow_all_headers and requested_headers is not None:
            headers["Access-Control-Allow-Headers"] = requested_headers
        elif requested_headers is not None:
            for header in [h.lower() for h in requested_headers.split(",")]:
                if header.strip() not in self.allow_headers:
                    failures.append("headers")

        # We don't strictly need to use 400 responses here, since its up to
        # the browser to enforce the CORS policy, but its more informative
        # if we do.
        if failures:
            failure_text = "Disallowed CORS " + ", ".join(failures)
            return PlainTextResponse(failure_text, status_code=400, headers=headers)

        return PlainTextResponse("OK", status_code=200, headers=headers)

    async def simple_response(
        self, scope: Scope, receive: Receive, send: Send, request_headers: Headers
    ) -> None:
        send = functools.partial(self.send, send=send, request_headers=request_headers)
        await self.app(scope, receive, send)

    async def send(
        self, message: Message, send: Send, request_headers: Headers
    ) -> None:
        if message["type"] != "http.response.start":
            await send(message)
            return

        message.setdefault("headers", [])
        headers = MutableHeaders(scope=message)
        headers.update(self.simple_headers)
        origin = request_headers["Origin"]
        has_cookie = "cookie" in request_headers

        # If request includes any cookie headers, then we must respond
        # with the specific origin instead of '*'.
        if self.allow_all_origins and has_cookie:
            headers["Access-Control-Allow-Origin"] = origin

        # If we only allow specific origins, then we have to mirror back
        # the Origin header in the response.
        elif not self.allow_all_origins and self.is_allowed_origin(origin=origin):
            headers["Access-Control-Allow-Origin"] = origin
            headers.add_vary_header("Origin")
        await send(message)



class TrustedHostMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        allowed_hosts: typing.Sequence[str] = None,
        www_redirect: bool = True,
    ) -> None:
        if allowed_hosts is None:
            allowed_hosts = ["*"]

        for pattern in allowed_hosts:
            assert "*" not in pattern[1:], ENFORCE_DOMAIN_WILDCARD
            if pattern.startswith("*") and pattern != "*":
                assert pattern.startswith("*."), ENFORCE_DOMAIN_WILDCARD
        self.app = app
        self.allowed_hosts = list(allowed_hosts)
        self.allow_any = "*" in allowed_hosts
        self.www_redirect = www_redirect

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self.allow_any or scope["type"] not in (
            "http",
            "websocket",
        ):  # pragma: no cover
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        host = headers.get("host", "").split(":")[0]
        is_valid_host = False
        found_www_redirect = False
        for pattern in self.allowed_hosts:
            if host == pattern or (
                pattern.startswith("*") and host.endswith(pattern[1:])
            ):
                is_valid_host = True
                break
            elif "www." + host == pattern:
                found_www_redirect = True

        if is_valid_host:
            await self.app(scope, receive, send)
        else:
            if found_www_redirect and self.www_redirect:
                url = URL(scope=scope)
                redirect_url = url.replace(netloc="www." + url.netloc)
                response = RedirectResponse(url=str(redirect_url))  # type: Response
            else:
                response = PlainTextResponse("Invalid host header", status_code=400)
            await response(scope, receive, send)



class HTTPSRedirectMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket") and scope["scheme"] in ("http", "ws"):
            url = URL(scope=scope)
            redirect_scheme = {"http": "https", "ws": "wss"}[url.scheme]
            netloc = url.hostname if url.port in (80, 443) else url.netloc
            url = url.replace(scheme=redirect_scheme, netloc=netloc)
            response = RedirectResponse(url, status_code=307)
            await response(scope, receive, send)
        else:
            await self.app(scope, receive, send)


#----------------------------- Data Security Service ----------------------------------



class EncryptMessage():
    ciphers:list = []
    #key_dir:str = '.key_dir'
    key_file_name:str = 'test_key.key'

    def generate_hash_key(self, key_name:str=None):
        hash_key = fn.generate_key()
        #print(hash_key)  
        
        #mkdir(key_dir)
        file = open(self.key_file_name, 'wb')
        file.write(hash_key)
        file.close()

    def generate_password_hash_key(self, plain_text_password:str):

        import base64
        import os
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        password_provided = plain_text_password # This is input in the form of a string
        password = password_provided.encode() # Convert to type bytes
        salt = b'salt_' # CHANGE THIS - recommend using a key from os.urandom(16), must be of type bytes
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password)) # Can only use kdf once
        return key 

    def encrypt_message(self, message):
        file = open(self.key_file_name, 'rb')
        key = file.read()
        f = fn(key)
        encrypted = f.encrypt(message.encode())
        print(encrypted)
        return encrypted

    def decrypt_message(self, encrypted_message):
        file = open(self.key_file_name, 'rb')
        key = file.read()
        f = fn(key)
        decrypted = f.decrypt(encrypted_message)
        
        file.close()
        return decrypted 


class EncryptFile:

    def encrypt_file():

        from cryptography.fernet import Fernet
        key = b'' # Use one of the methods to get a key (it must be the same when decrypting)
        input_file = 'test.txt'
        output_file = 'test.encrypted'

        with open(input_file, 'rb') as f:
            data = f.read()

        fernet = Fernet(key)
        encrypted = fernet.encrypt(data)

        with open(output_file, 'wb') as f:
            f.write(encrypted)

        # You can delete input_file if you want

    def decrypt_file():
        from cryptography.fernet import Fernet
        key = b'' # Use one of the methods to get a key (it must be the same as used in encrypting)
        input_file = 'test.encrypted'
        output_file = 'test.txt'

        with open(input_file, 'rb') as f:
            data = f.read()

        fernet = Fernet(key)
        encrypted = fernet.decrypt(data)

        with open(output_file, 'wb') as f:
            f.write(encrypted)

        # You can delete input_file if you want

#----------------------------- ID Generation Service ----------------------------------

class GenerateId:
    tags = dict(
            doc='[h-z5-9]{8:16}',
            app='[a-z0-9]{16:32}',
            key='[a-z0-9]{32:32}',
            job='[a-j0-7]{8:8}',
            user='[0-9]{4:6}',
            item='[a-n1-9]{8:8}',
            code='[a-x2-8]{24:32}'
        )
        
    async def genid(self, doc_tag:str=None):
        """ 
            Doc Tags: String( doc, app, key, job, user, item, code,task,name)
            UseCase: 
                        >>> import genny
                        >>> from genny import genid
                        >>> from genny import genid as gi
                        
                        >>> id = genny.genid('user')
                        >>> id = genid('user')
                        >>> id = gi('user')
                Yeilds ... U474390
                        ... U77301642
                        ... U1593055
        
        """
        
        if doc_tag == 'user':
            #u_id = StringGenerator(str(self.tags[doc_tag])).render(unique=True)
            return f"U{StringGenerator(str(self.tags[doc_tag])).render(unique=True)}"
        return StringGenerator(str(self.tags[doc_tag])).render(unique=True)
            

    async def nameid(self, fn:str='Jane',ln:str='Dear',sec:int=5):
        """ 
            Name Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import nameid
                        >>> from genny import nameid as nid
                        
                        >>> id = genny.nameid('Peter','Built',6)
                        >>> id = nameid('Peter','Built',5)
                        >>> id = nid('Peter','Built',4)
                        >>> id = nid() # default false id 
                        
                Yeilds ... PB474390
                        ... PB77301
                        ... PB1593
                        ... JD1951
        
        """
        code = '[0-9]{4:%s}'% int(sec)
        return f"{fn[0].capitalize()}{ln[0].capitalize()}{StringGenerator(str(code)).render(unique=True)}"
               

    async def short_nameid(self, fn:str='Jane',ln:str='Dear',sec:int=2):
        """ 
            Name Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import short_nameid
                        >>> from genny import short_nameid as id
                        
                        >>> id = genny.short_nameid('Peter','Built',2)
                        >>> id = short_nameid('Peter','Built')
                        >>> id = id(p','b',3)
                        >>> id = id() # default false id 
                        
                Yeilds ... PB47
                        ... PB54
                        ... PB69
                        ... JD19
        
        """
        code = '[0-9]{2:%s}'% int(sec)
        return f"{fn[0].capitalize()}{ln[0].capitalize()}{StringGenerator(str(code)).render(unique=True)}"
        

    async def eventid(self, event,event_code,sec=8):
        """EventId 
            Event Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import eventid
                        >>> from genny import eventid as id
                        
                        >>> id = genny.eventid('Product','LAUNCH',6)
                        >>> id = eventid('Product','LAUNCH',5)
                        >>> id = id('Product', 'LAUNCH',4)                       
                Yeilds ... PROLAUNCH-884730
                        ... PROLAUNCH-18973
                        ... PROLAUNCH-4631                       
        
        """
        code = '[0-9]{4:%s}'% int(sec)
        return f"{event[:3].upper()}{event_code}-{StringGenerator(str(code)).render(unique=True)}"
        

    async def short_eventid(self, event,event_code,sec=2):
        """ShortEventId 
            Event Identification by initials fn='Jane', ln='Dear' and given number sequence sec=2.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import shorteventid
                        >>> from genny import shorteventid as id
                        
                        >>> id = genny.shorteventid('Product','LAUNCH',2)
                        >>> id = shorteventid('Product','LAUNCH')
                        >>> id = id('Product', 'LAUNCH',3)
                Yeilds ... PROLAUNCH-88
                        ... PROLAUNCH-90
                        ... PROLAUNCH-461                       
        
        """
        code = '[0-9]{2:%s}'% int(sec)
        return f"{event[:3].upper()}{event_code}-{StringGenerator(str(code)).render(unique=True)}"
        
        
    def gen_id(self, doc_tag:str=None):
        """ 
            Doc Tags: String( doc, app, key, job, user, item, code,task,name)
            UseCase: 
                        >>> import genny
                        >>> from genny import genid
                        >>> from genny import genid as gi
                        
                        >>> id = genny.genid('user')
                        >>> id = genid('user')
                        >>> id = gi('user')
                Yeilds ... U474390
                        ... U77301642
                        ... U1593055
        
        """
        
        if doc_tag == 'user':
            #u_id = StringGenerator(str(self.tags[doc_tag])).render(unique=True)
            return f"U{StringGenerator(str(self.tags[doc_tag])).render(unique=True)}"
        return StringGenerator(str(self.tags[doc_tag])).render(unique=True)
            

    def name_id(self, fn:str='Jane',ln:str='Dear',sec:int=5):
        """ 
            Name Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import nameid
                        >>> from genny import nameid as nid
                        
                        >>> id = genny.nameid('Peter','Built',6)
                        >>> id = nameid('Peter','Built',5)
                        >>> id = nid('Peter','Built',4)
                        >>> id = nid() # default false id 
                        
                Yeilds ... PB474390
                        ... PB77301
                        ... PB1593
                        ... JD1951
        
        """
        code = '[0-9]{4:%s}'% int(sec)
        return f"{fn[0].capitalize()}{ln[0].capitalize()}{StringGenerator(str(code)).render(unique=True)}"
               

    def short_name_id(self, fn:str='Jane',ln:str='Dear',sec:int=2):
        """ 
            Name Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import short_nameid
                        >>> from genny import short_nameid as id
                        
                        >>> id = genny.short_nameid('Peter','Built',2)
                        >>> id = short_nameid('Peter','Built')
                        >>> id = id(p','b',3)
                        >>> id = id() # default false id 
                        
                Yeilds ... PB47
                        ... PB54
                        ... PB69
                        ... JD19
        
        """
        code = '[0-9]{2:%s}'% int(sec)
        return f"{fn[0].capitalize()}{ln[0].capitalize()}{StringGenerator(str(code)).render(unique=True)}"
        

    def event_id(self, event,event_code,sec=8):
        """EventId 
            Event Identification by initials fn='Jane', ln='Dear' and given number sequence sec=5.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import eventid
                        >>> from genny import eventid as id
                        
                        >>> id = genny.eventid('Product','LAUNCH',6)
                        >>> id = eventid('Product','LAUNCH',5)
                        >>> id = id('Product', 'LAUNCH',4)                       
                Yeilds ... PROLAUNCH-884730
                        ... PROLAUNCH-18973
                        ... PROLAUNCH-4631                       
        
        """
        code = '[0-9]{4:%s}'% int(sec)
        return f"{event[:3].upper()}{event_code}-{StringGenerator(str(code)).render(unique=True)}"
        

    def short_event_id(self, event,event_code,sec=2):
        """ShortEventId 
            Event Identification by initials fn='Jane', ln='Dear' and given number sequence sec=2.
            
            UseCase: 
                        >>> import genny
                        >>> from genny import shorteventid
                        >>> from genny import shorteventid as id
                        
                        >>> id = genny.shorteventid('Product','LAUNCH',2)
                        >>> id = shorteventid('Product','LAUNCH')
                        >>> id = id('Product', 'LAUNCH',3)
                Yeilds ... PROLAUNCH-88
                        ... PROLAUNCH-90
                        ... PROLAUNCH-461                       
        
        """
        code = '[0-9]{2:%s}'% int(sec)
        return f"{event[:3].upper()}{event_code}-{StringGenerator(str(code)).render(unique=True)}"
        
        
        
 
#generate_id = GenerateId()
#encrypt_message_file = EncryptMessage()

