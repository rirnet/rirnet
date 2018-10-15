import os
import bcolors

folderpath = "../code"
word = '#TODO'
dont_search_file = 'print_todo.py'
dont_search_folders = '../code/__pycache__' #'./__pycache__',


def get_between(s, first):
    try:
        start = s.index( first ) + len( first )
        return str(s[start:])
    except ValueError:
        return ""

TODO_count = 0
for(path, dirs, files) in os.walk(folderpath, topdown=True):
    if(path != dont_search_folders):
        for filename in files:
            if(filename != dont_search_file):
                filepath = os.path.join(path, filename)
                with open(filepath, 'r') as currentfile:
                    line_count = 1
                    for line in currentfile:
                        if word in line:
                            print(
                                    bcolors.CRED + bcolors.CBOLD +
                                    filename +
                                    bcolors.CEND +
                                    ' at line: ' +
                                    bcolors.CRED + bcolors.CBOLD +
                                    str(line_count) + '\n' +
                                    bcolors.CEND +
                                    bcolors.CITALIC +
                                    get_between(line, '#TODO') +
                                    bcolors.CEND)
                            TODO_count += 1
                        line_count += 1

print('Found ' + bcolors.BOLD + bcolors.CRED + str(TODO_count) + bcolors.CEND  + ' TODO:s')
