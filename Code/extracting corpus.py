
# EXTRACTING TEXT CORPORA

import re, sys

def is_time_stamp(l):
  if l[:2].isnumeric() and l[2] == ':':
    return True
  return False

def has_letters(line):
  if re.search('[a-zA-Z]', line):
    return True
  return False

def has_no_text(line):
  l = line.strip()
  if not len(l):
    return True
  if l.isnumeric():
    return True
  if is_time_stamp(l):
    return True
  if l[0] == '(' and l[-1] == ')':
    return True
  if not has_letters(line):
    return True
  return False

def is_lowercase_letter_or_comma(letter):
  if letter.isalpha() and letter.lower() == letter:
    return True
  if letter == ',':
    return True
  return False

def clean_up(lines):
  new_lines = []
  for line in lines[1:]:
    if has_no_text(line):
      continue
    elif len(new_lines) and is_lowercase_letter_or_comma(line[0]):
      #combine with previous line
      new_lines[-1] = new_lines[-1].strip() + ' ' + line
    else:
      #append line
      new_lines.append(line)
  return new_lines

df = pd.DataFrame(columns=['lines'])

def main(filename):
  lst=[]
  file_name = filename
  file_encoding = 'utf-8'
  with open(file_name, encoding=file_encoding, errors='replace') as f:
    lines = f.readlines()
    new_lines = clean_up(lines)

  for line in new_lines:
      word= line.strip()
      word = word.strip('=-')
      lst.append(word)

  d = pd.DataFrame(lst)
  return d






# import OS module
import os

# Get the list of all files and directories
path = "book4/"
dir_list = os.listdir(path)
book1 = pd.DataFrame()
# prints all files
for i in dir_list:
    df = main(path+i)
    book1  = pd.concat([book1,df],ignore_index=True)

print(book1.dropna)
book1.to_csv('book4.csv',index=False)


df1 = pd.read_csv('book1.csv')
df2= pd.read_csv('book2.csv')
df3 = pd.read_csv('book3.csv')
df4 = pd.read_csv('book4.csv')

korra = pd.concat([df1,df2,df3,df4],ignore_index=True)
korra.to_csv('korra.csv',index=False)

