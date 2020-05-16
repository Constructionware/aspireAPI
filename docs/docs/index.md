# Aspire&reg;

Aspire is a lightweight fully production ready pure Python ASGI web application framework and tool belt, built with the latest cool web technologies to make creating and deploying simple to large scale complex web applications less difficult.<br>


    from aspire import API  

    app = API() # Initialize an Application

    @app.route('/')
    def index(req, response):
        response.text = "Hello, World!"

    if __name__ == '__main__':
        app.run()

``` 
    $ aspire serve appName
    * Serving Aspire Application appName
    * Running on http://127.0.0.1:1090/     (CTRL + C  to quit)

```

 A fork of Kenneth Reitz Responder | Inspired by Pallets Flask  and Tom Christie,s Starlette

For full documentation visit [mkdocs.org](https://www.mkdocs.org).



## Cli

* `mkdir my-cool-projects` - Create a new project directory.
* `cd my-cool-projects` - home/my-cool-projects/
* `aspire create_api myCoolApi` - Create a new skeleton REST Api called myCoolApi.
* `aspire create_app myCoolApp` - Create a new skeleton Web project called myCoolApp.
* `cd myCoolApi` - home/my-cool-projects/myCoolApi/
* `cd myCoolApp` - home/my-cool-projects/myCoolApp/
* `aspire serve` - Start the server in development mode with live-reloading and dev tools.
* `aspire build` - Build the a deployable production ready application with dev mode off.
* `aspire run` - Runs a Built application in Production mode on the Host Machine.
* `aspire -h` - For Help.
* `aspire --v` - Current version 

aspire default to port 1090 | point your browser at <a href="http://localhost:1090">localhost:1090</a> 

## Project layout



    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
