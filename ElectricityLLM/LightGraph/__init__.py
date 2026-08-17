from .GraphGet import full_search


def run_other_pipeline(*args, **kwargs):
    from .document_deal import run_other_pipeline as _run_other_pipeline
    return _run_other_pipeline(*args, **kwargs)
