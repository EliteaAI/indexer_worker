class InternalSDKError(Exception):
    pass


try:
    from elitea_sdk.runtime.exceptions import PipelineConfigurationError
except ImportError:
    class PipelineConfigurationError(Exception):
        pass
