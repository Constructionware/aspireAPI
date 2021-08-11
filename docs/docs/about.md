# About Aspire

The Motivation behind rolling our own server customised with all the niceties of
getting complex tasks done in a simple, productive and reliable way arose from failing dependencies nightmare
as a result of developers applying updates to their software's dependencies late or un-maintained packages . 

Aspire seeks to address this issue by relying heavily on pure python system modules implementation 
and packaging a vast portion of its dependencies on board, kind of like "Batteries included".
Packaged dependencies are well curated removing any unused or heavy imports, redundant codes and unesseary bloat,
optimising and improving code quality.

```
    import os
    
    def get_path():
        path = os.path.abspath(__file__)
        return path
```

In the above implementation the entire "os" moule is imported into memory, this is repeated multiple times throught some code base.
resulting in unnessesary cpu loads, a heavier memory usage and eventually slower code performance.

```
    from os.path import abspath

    def get_path():
        return abspath(__file__)
```

Optimising our dependents We dropped the entire os import and opted for a tiny portion with a direct import of its sub module.
we depricated the memory waste variable "path" and return the result directly. We get a faster executing code with vastly reduced memory footprint. 

The result is a much more agile, stable and reliable application with a longer operable lifespan between updates and 
an apperciated reduction in memory footprint both on the host and in runtime.

