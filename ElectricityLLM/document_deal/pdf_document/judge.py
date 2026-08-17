def judgement(file_name:str):
    if file_name == None:
        return "null"
    lower_file_name = file_name.lower()
    if ".pdf" in lower_file_name:
        return "pdf"
    elif ".docx" in lower_file_name:
        return "docx"
    elif ".doc" in lower_file_name:
        return "doc"
    else:
        return "unsupported"
