import os

folderpath = "."
word = '#TODO'
dont_search_file = 'print_todo.py'
dont_search_folder = './__pycache__'

def get_between(s, first):
    try:
        start = s.index( first ) + len( first )
        return str(s[start:])
    except ValueError:
        return ""

TODO_count = 0
for(path, dirs, files) in os.walk(folderpath, topdown=True):
    if(path != dont_search_folder):
        for filename in files:
            if(filename != dont_search_file):
                filepath = os.path.join(path, filename)
                with open(filepath, 'r') as currentfile:
                    line_count = 1
                    for line in currentfile:
                        if word in line:
                            print(filename + ' at line: '
                                    + str(line_count) + '\n' +
                                    get_between(line, '#TODO'))
                            TODO_count += 1
                        line_count += 1

print('Found ' + str(TODO_count) + ' TODO:s')


