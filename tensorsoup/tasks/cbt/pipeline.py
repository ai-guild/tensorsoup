def remove_chars(text, chars='_.,!#$%^&*()\?:{}[]/;`~*+|'):
    return ''.join([ ch for ch in text if ch not in chars ])

def strip(text):
    return text.replace('  ', ' ').strip().lower()

def replace(text):
    return strip(text.replace('-', ' ').replace(" n't", "n't"))#.
    #replace("''", ' '))

def has_alnum(w):
    for ch in w:
        if ch.isalnum():
            return w
    return ''

def remove_words(text):
    modtext = ' '.join([ w for w in text.split(' ') if has_alnum(w) ])
    return modtext if modtext else '<unk>'

text_pipeline = lambda text : remove_words(replace(remove_chars(text)))
word_pipeline = lambda w : replace(remove_chars(w))

def fix_joints(rtext):
    
    def fix_joint(w):
        try:
            pre, post = w.split('\n')
        except:
            print(w.split('\n'))
        return has_alnum(pre) + '\n' + has_alnum(post)
    
    text = []
    for w in rtext.split(' '):
        if '\n' in w:
            if '\n\n' not in w:
                w = fix_joint(w)
        text.append(w)
            
    return ' '.join(text)

rtext_pipeline = lambda rtext :  remove_words(replace(remove_chars(
    fix_joints(rtext), '_.,!#$%^&*()\?:{}[];`~*+')))
