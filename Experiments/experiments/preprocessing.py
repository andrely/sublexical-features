import re


def mahoney_clean(str):
    str = " " + str + " "
    str = str.lower()

    str = re.sub('1', ' one ', str, flags=re.MULTILINE)
    str = re.sub('2', ' two ', str, flags=re.MULTILINE)
    str = re.sub('3', ' three ', str, flags=re.MULTILINE)
    str = re.sub('4', ' four ', str, flags=re.MULTILINE)
    str = re.sub('5', ' five ', str, flags=re.MULTILINE)
    str = re.sub('6', ' six ', str, flags=re.MULTILINE)
    str = re.sub('7', ' seven ', str, flags=re.MULTILINE)
    str = re.sub('8', ' eight ', str, flags=re.MULTILINE)
    str = re.sub('9', ' nine ', str, flags=re.MULTILINE)
    str = re.sub('0', ' zero ', str, flags=re.MULTILINE)

    str = re.sub('[^a-z]', ' ', str, flags=re.MULTILINE)
    str = re.sub('\s+', ' ', str, flags=re.MULTILINE)

    return str