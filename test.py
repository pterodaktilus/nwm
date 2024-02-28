import re
s = """doc. Josef Tyl, Rudolf Srp, Ph.D., Pavel Vlk, doc. RNDr. Petr Berka, Ph.D., Jan Hora"""
print(re.findall(r'''((?:(?:(?:[Pp]rof.)|(?:[Dd]oc.)) ) .*(, Ph\.D\.)?)''',s)) #