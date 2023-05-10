import os

import gin

gin.add_config_file_search_path(os.path.dirname(__file__))
gin.add_config_file_search_path(
    os.path.join(os.path.dirname(__file__), 'configs'))

for subdir in ["arch", "condition", "type"]:
    gin.add_config_file_search_path(
        os.path.join(os.path.dirname(__file__), 'configs', subdir))
