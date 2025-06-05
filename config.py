# Paths
IMG_FOLDER = 'images/'
EX_IMG_FOLDER = 'example_images/'
DATA_FOLDER = 'data/'
LOG_FOLDER = 'log/'
QUESTION_FILE = '12questions.pkl'
# Parameters
RESPONSE_KEYS = ('g', 'h')  # 'g' for correct/likely, 'h' for incorrect/unlikely
IMG_MIN_WAIT = 3  # Minimum time to show each image in seconds
FIXATION_TIME = 1.0
RESPONSE_TIME = 2.5
# Feedbacks
FEEDBACK_RIGHT = 'Correct!'
FEEDBACK_WRONG = 'WRONG'
FEEDBACK_SLOW = 'Too slow. Please respond faster.'
RED = '#ff0000'
GREEN = '#84ff84'
BLACK = '#000000'

# Instructions
INSTR_BEGIN = [
    'Welcome! In this experiment, you will look at images and answer questions about them.',
    'Each block of images will feature a different type of question.',
    'For each image, you will first see a question, followed by the image, and then a potential answer to the question.\n\n' \
        'The answer will either be correct/likely or incorrect/unlikely.',
    f'You can view the image for as long as you like, but no less than {int(IMG_MIN_WAIT)} seconds. ' \
        'When you\'re ready, press SPACE to dismiss the image and see the potential answer.',
    'Your task is to decide whether the answer is correct/likely or incorrect/unlikely by pressing G or H on the keyboard.',
    'Importantly, the answers will be shown very briefly, so please pay close attention while viewing the image.\n\n' \
        'Make sure you have a correct answer in mind while viewing the image, and then respond as quickly as possible.',
    'In the following examples, you will be asked about shapes in the images.\n\n',
]
INSTR_EXP = 'Great job completing the examples!\n\n' \
            'If you have any questions, please ask the experimenter now.\n\n' \
            'If not, please make sure you are comfortable and ready to start the experiment.'
INSTR_BLOCKS = 'In the following block, you will be asked {}.\n\n' \
               'Again, the answers will be shown very briefly, so please pay close attention while viewing the image ' \
               'and have a correct answer in mind, then respond as quickly as possible.'
INSTR_BLOCK_THEME = {
    'fact-ppl': 'about facts in the images',
    'fact-nonppl': 'about facts in the images',
    'inference-social': 'about people\'s actions or intentions',
    'inference-nonsocial': 'to infer the time, weather, or environmental conditions',
    'next': 'to predict the next event',
    'relation': 'to infer relationships between people',
    'feel': 'to infer people\'s feelings or emotions',
}

EXAMPLE_QUESTIONS = {
    'ex1.jpg': {
        'question': 'What is the shape of the mountains in the background?',
        'answer': 'The mountains are round and flat on the top\n\n\n' \
                  'Press G or H',
        'correct': 'h',
    },
    'ex2.jpg': {
        'question': 'What is the shape of the bookself?',
        'answer': 'The bookshelf is triangular',
        'correct': 'h',
    },
    'ex3.jpg': {
        'question': 'What is the shape of the sun in the sky?',
        'answer': 'The sun is a perfect circle',
        'correct': 'g',
    }
}

INSTR_END = 'This is the end of the experiment.\n\nThank you for your participation!'
