"""
Compatibility layer for erlport to work with newer Python versions.
This fixes the issue with getargspec being removed in Python 3.11+
"""

import sys
import inspect

# Add compatibility for getargspec which was removed in Python 3.11
if not hasattr(inspect, 'getargspec'):
    print("Adding compatibility for getargspec in Python 3.11+")
    
    # Create a compatibility function using getfullargspec
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return inspect.ArgSpec(
            args=spec.args,
            varargs=spec.varargs,
            keywords=spec.varkw,
            defaults=spec.defaults
        )
    
    # Add it to the inspect module
    inspect.getargspec = getargspec
    
    print("Compatibility layer for erlport installed successfully")
