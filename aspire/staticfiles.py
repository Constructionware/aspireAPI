import typing

from aspire.utils.staticfiles import StaticFiles


class StaticFiles(StaticFiles):
    
    """Pending issues    
    https://github.com/encode/starlette/issues/625
    
    https://github.com/encode/starlette/pull/626
    """
    all_directories:list = None
    

    def add_directory(self, directory: str) -> None:
        self.all_directories = [*self.all_directories, *self.get_directories(directory)]

