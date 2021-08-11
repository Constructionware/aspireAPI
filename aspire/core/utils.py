# aspire.core.utils.py | Aspire Security Services | Csware | Aug 11 2021 | MIT



from genericpath import isfile
from os import path, getlogin, mkdir, remove

# Use SimpleJson if available , resorts to json if not
try: 
    import simplejson as json
except ImportError:
    import json

try:
    import  aspire.core.status  as status
except ImportError:
    pass

class JsonCRURD:
    """ Creat Read Update Rollback Delete Json Files"""

    # CREATE 
    async def write_json(self, data: dict=None, handle: str=None):
        """ Creates a new file on the file system """
        try:
            payload = json.dumps(data, indent = 4) # serialize data
            del(data)
            with open(path.join(self.file_dir, handle), "w") as outfile:
                outfile.write(payload)
            del(payload)
            del(handle)
            del(outfile)
            return status.HTTP_201_CREATED
            # adding .json suffix to file
        except:
            return status.HTTP_400_BAD_REQUEST

    # READ
    async def read_json(self, handle: str=None):
        ''' Reads a json file from the system'''
        try:
            # Reading a .json            
            with open(path.join(self.file_dir, handle), "r") as infile:
                json_object = json.load(infile)                    
            del(handle)
            del(infile)
            return json_object            
        except Exception as e: 
            return {'status':status.HTTP_404_NOT_FOUND, "error": e}
    # UPDATE
    async def update_json(self,data, handle: str=None):
        ''' Reads a json file from the system'''
        try:
            # keep old data in cache in case of rollback
            self.old_data = await self.read_json(handle=handle)
            new_data = self.old_data | data
            await self.write_json(new_data, handle=handle)        
            return status.HTTP_200_OK
        except Exception as e:
            return e

    # ROLLBACK
    async def rollback_json_update(self, handle: str=None):
        ''' Rolls back last 100 updates '''
        try:
            await self.write_json(self.old_data , handle=handle) 
            del(self.old_data) # removes the restored data from cache       
            return status.HTTP_200_OK
        except Exception as e:
            return e

    # DELETE
    async def delete_json(self, handle: str=None):
            ''' Deletes a file ''' 
            try:
                remove(path.join(self.file_dir, handle) )
                del(handle)        
                return status.HTTP_200_OK
            except Exception as e:
                return e



class FileWriter(
    JsonCRURD

    ):
    """ File writing Utility
    Creates a hidden file directory called asp_bak in the users Home directory 
    """
    file_path = path.abspath(f'C:/Users/{getlogin()}')
    file_dir = path.join(file_path, '.asp_bak')

    def __init__(self):
        self.setup_file_dir()

    # checking for file_dir
    def setup_file_dir(self): 
        """ Creates a new directory on the file system"""       
        if path.isdir(self.file_path):
            pass
        else:
            mkdir(self.file_dir)

