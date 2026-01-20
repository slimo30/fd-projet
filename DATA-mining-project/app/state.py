class GlobalState:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalState, cls).__new__(cls)
            cls._instance.df = None
            cls._instance.current_filepath = None
            cls._instance.clustering_results = {}
            cls._instance.ml_results = {}
        return cls._instance

# Create a singleton instance to be imported
state = GlobalState()
