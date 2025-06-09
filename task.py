#!/usr/bin/env python


import pygaze
from pygaze import eyetracker, libscreen
import random
import os
import pickle
from psychopy_util import *
from config import *


def show_one_trial(presenter, image, img_bg, question, feedbacks, trialname, eyetracker=None):
    presenter.show_fixation(FIXATION_TIME)
    # show instruction
    presenter.show_instructions(question['question'])
    if eyetracker:
        eyetracker.start_recording()
        eyetracker.log(trialname + ' Showing fixation')
    presenter.show_fixation(FIXATION_TIME)
    # show image
    image.pos = presenter.CENTRAL_POS
    if eyetracker:
        eyetracker.log(trialname + ' Showing image ' + image._imName)
    presenter.draw_stimuli_for_duration([img_bg, image], duration=IMG_MIN_WAIT)
    response = presenter.draw_stimuli_for_response([img_bg, image], 'space')
    if eyetracker:
        eyetracker.log(trialname + ' End of image display ' + image._imName)
        eyetracker.stop_recording()
    # show answer and get response
    ans_stim = visual.TextStim(presenter.window, text=question['answer'], pos=presenter.CENTRAL_POS, wrapWidth=1.5)
    response = presenter.draw_stimuli_for_response(ans_stim, RESPONSE_KEYS, max_wait=RESPONSE_TIME)
    data = {
        'response': response,
        'response_correct': (response[0] == question['correct']) if response else None
    }
    data.update(question)
    # show feedback
    feedback_stims = None
    if response is None:
        feedback_stims = feedbacks[1]
    else:
        feedback_stims = feedbacks[0][int(data['response_correct'])]
    presenter.draw_stimuli_for_duration(feedback_stims, duration=1)
    return data


def validation(items):
    # check for empty fields
    for key in items.keys():
        if items[key] is None or len(items[key]) == 0:
            return False, str(key) + ' cannot be empty.'
    # everything is okay
    return True, ''


def main():
    # subject ID dialog
    sinfo = {'ID': '', 'Part': ['1', '2', '3'], 'Fullscreen': ['Yes', 'No']}
    show_form_dialog(sinfo, validation, order=['ID', 'Fullscreen'])
    sid = int(sinfo['ID'])
    part = int(sinfo['Part'])
    random.seed(sid)

    # create logging file
    infoLogger = DataLogger(LOG_FOLDER, f'{sid}pt{part}.log', 'info_logger', logging_info=True)
    # create data file
    dataLogger = DataLogger(DATA_FOLDER, f'{sid}pt{part}.txt', 'data_logger')
    # save info from the dialog box
    dataLogger.write_json({
        k: str(sinfo[k]) for k in sinfo.keys()
    })
    # create window
    presenter = Presenter(fullscreen=(sinfo['Fullscreen'] == 'Yes'), info_logger='info_logger')

    # load data
    images = presenter.load_all_images(IMG_FOLDER, '.png')
    with open(os.path.join(os.getcwd(), QUESTION_FILE), 'rb') as f:
        questions = pickle.load(f)
    ex_images = presenter.load_all_images(EX_IMG_FOLDER, '.jpg')

    # image background
    black_bg = visual.Rect(presenter.window, width=2.1, height=2.1, fillColor=BLACK)
    # feedback stimuli
    correct_feedback = visual.TextStim(presenter.window, FEEDBACK_RIGHT, color=GREEN)
    incorrect_feedback = visual.TextStim(presenter.window, FEEDBACK_WRONG)
    incorrect_bg = visual.Rect(presenter.window, width=2.1, height=2.1, fillColor=RED)
    resp_feedback = ([incorrect_bg, incorrect_feedback], correct_feedback)
    no_resp_feedback = [incorrect_bg, visual.TextStim(presenter.window, FEEDBACK_SLOW)]

    # set up eyetracker
    pygaze.settings.DISPTYPE = 'psychopy'
    pygaze.expdisplay = presenter.window
    tracker = eyetracker.EyeTracker(libscreen.Display(), trackertype='eyetribe', logfile=f'log/{str(sid)}pt{part}_eyetribe') # 'eyetribe'

    # randomization
    imglist = list(questions.keys())
    blocks1 = [k for k in INSTR_BLOCK_THEME.keys() if not k.startswith('fact')]
    blocks2 = [k for k in INSTR_BLOCK_THEME.keys() if k.startswith('fact')]
    random.shuffle(blocks1)
    random.shuffle(blocks2)
    blocks = blocks1 + blocks2  # fact blocks at the end
    if part == 1:
        blocks = blocks[:2]
    elif part == 2:
        blocks = blocks[2:4]
    elif part == 3:
        blocks = blocks[4:]
    
    if part == 1:
        # show instructions
        for instr in INSTR_BEGIN:
            presenter.show_instructions(instr)
        # show examples
        for t, img in enumerate(EXAMPLE_QUESTIONS):
            data = show_one_trial(presenter, ex_images[img], black_bg, EXAMPLE_QUESTIONS[img], (resp_feedback, no_resp_feedback), f'example{t}', tracker)
            # log example trial data
            infoLogger.logger.info('Writing to data file')
            dataLogger.write_json({'example_trial_index': t, 'response': data})

    # show experiment
    presenter.show_instructions(INSTR_EXP)
    for b in blocks:
        random.shuffle(imglist)
        presenter.show_instructions(INSTR_BLOCKS.format(INSTR_BLOCK_THEME[b]))
        for t, img in enumerate(imglist):
            data = show_one_trial(presenter, images[img], black_bg, questions[img][b], (resp_feedback, no_resp_feedback), f'{b}_{t}', tracker)
            data.update({'block': b, 'trial_index': t, 'image': img})
            infoLogger.logger.info('Writing to data file')
            dataLogger.write_json(data)

    # end of experiment
    presenter.show_instructions(INSTR_END)
    infoLogger.logger.info('End of experiment')


if __name__ == '__main__':
    main()
