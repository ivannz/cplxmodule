from .views import fix_dim, complex_view, window_view


# i am lazy to rewrite code, so here is a factory
def torch_module(fn, base, name=None):
    # a class template
    class template_(*base):
        def forward(self, input):
            return fn(input)

    # update the name and qualname
    name_ = name if name is not None else fn.__name__.title()
    setattr(template_, "__name__", name_)
    setattr(template_, "__qualname__", name_)

    # update defaults
    for attr in ('__module__', '__doc__'):
        try:
            value = getattr(fn, attr)
        except AttributeError:
            pass
        else:
            setattr(template_, attr, value)
    # end for

    return template_