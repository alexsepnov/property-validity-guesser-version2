import re

string1 = "jln. karang tinggal no.14"
string2 = "AA, BB,CC DD EE FF ,HH"

string1x = re.sub(r"^(jl. |jln. |jl.|jln.)", "", string1)
print string1x

special_word = ['AA', 'BB', 'CC', 'EE', 'HH']
print string2
string2x  = re.split(', | ,|,|\s', string2)
print len(string2x)
for str in string2x:
    print str
    if str in special_word:
        print "ada"


string3 = "jln, karang tinggal no.14"
string_array = re.split(', |; |\. | ,| ;| \.|,|;|\.|\s', string3)
print string_array


import os
data_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
print data_path
data_path = os.path.realpath(os.path.dirname(os.path.realpath(__file__)) + '/../data')
print data_path
