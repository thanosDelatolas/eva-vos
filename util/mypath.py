import socket
import getpass

class Path:
    
    @staticmethod
    def db_root_path(dataset='DAVIS_17'):
        # if socket.gethostname().__contains__('-gpu') or socket.gethostname() == 'titans':
        #     return f'/scratch/{getpass.getuser()}/data/{dataset}'
        # else : 
        #     return f'data/{dataset}'
        return f'data/{dataset}'
    
    @staticmethod
    def data_path():
        # if socket.gethostname().__contains__('-gpu') or socket.gethostname() == 'titans':
        #     return f'/scratch/{getpass.getuser()}/data/'
        # else : 
        #     return f'data/'
        return 'data/'
        
        
    
