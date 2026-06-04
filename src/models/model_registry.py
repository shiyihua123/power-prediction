_MODEL_REGISTRY = {}

def register_model(name):
    """装饰器：把模型类注册到注册表中"""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name, config, model_config=None):
    """根据名字和参数创建模型实例"""
    if name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(f"未知模型: '{name}'。当前可用模型: {available}")
    return _MODEL_REGISTRY[name](config, model_config)
