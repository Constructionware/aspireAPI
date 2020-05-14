from aspire import API

from aspire.utils.cors_helper import CORSMiddleware
from aspire.utils.routing import Route, Mount
from aspire.utils.responses import JSONResponse
from aspire.utils.templating import Jinja2Templates
#from utils.staticfiles import StaticFiles


# SETUP

host='0.0.0.0'
port=8600
cert = ".ssl/app.crt"
pem=".ssl/app.pem"
admin_email = "worksman.io@gmail.com"
templates = Jinja2Templates(directory='templates')


app = API(
    title='The Worksman Api',
    version="1.0.4",
    static_dir='templates/static',
    templates_dir= 'templates',
    static_route='/static',
    secret_key="eax554ert",#genny.genid('app'),
    auto_escape=True ,
    debug= True
)


# CONFIGURATION
#app.state.ADMIN_EMAIL = admin_email
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    allow_origin_regex=None,
    expose_headers=[
       "Access-Control-Allow-Origin",
       "Access-Control-Allow-Credentials",
       "Access-Control-Allow-Expose-Headers"
    ],
    max_age=5600
)

#@app.exception_handler(404)
async def not_found(request, exc):
    """ Return JSON Not Found Stamp"""
    context = {"request": request}
    return  JSONResponse({ "status_code": 404})


# Site Web Manifest
#@app.route('/manifest.json')
#def web_manifest( req, resp ):
 #   resp.headers["Content-Type"] = "text/cache=manifest"    
  #  resp.html = app.template('manifest.json') 

# Worsman Client

# Application Client Route
@app.route('/')
@app.route('/{path}')
def index(req, resp, *, path=""):
    resp.html = app.template('index.html')


#app.mount('/api', wkmapi.api)
#app.mount('/img', image_api.api)
                          
'''@atexit.register
def shutdown():
    print('disconnecting  DBMS.....')
    dbms.disconnect()'''

app.run(
    
    host=host,
    port=port,
    limit_concurrency=10000,
    limit_max_requests=1000,
    loop='asyncio',
    # reload=True,
    access_log=False
)