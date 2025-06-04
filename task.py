#!/usr/bin/env python

# TODO Black background for image?
# TODO make image size the same
# TODO add eyetracker

from pygaze import eyetracker
import random
import os
import pickle
from psychopy_util import *
from config import *


def show_one_trial(presenter, image, img_bg, questions, feedbacks):
    presenter.show_fixation(FIXATION_TIME)
    # show instruction
    presenter.show_instructions(questions['question'])
    presenter.show_fixation(FIXATION_TIME)
    # show image
    image.pos = presenter.CENTRAL_POS
    presenter.draw_stimuli_for_duration([img_bg, image], duration=IMG_MIN_WAIT)
    response = presenter.draw_stimuli_for_response([img_bg, image], 'space')
    # show answer
    ans_stim = visual.TextStim(presenter.window, text=questions['answer'], pos=presenter.CENTRAL_POS, wrapWidth=1.5)
    response = presenter.draw_stimuli_for_response(ans_stim, RESPONSE_KEYS, max_wait=RESPONSE_TIME)
    # show feedback
    feedback_stims = None
    if response is None:
        feedback_stims = feedbacks[1]
    else:
        feedback_stims = feedbacks[0][int(response[0] == questions['correct'])]
    presenter.draw_stimuli_for_duration(feedback_stims, duration=1)
    return response


def validation(items):
    # check for empty fields
    for key in items.keys():
        if items[key] is None or len(items[key]) == 0:
            return False, str(key) + ' cannot be empty.'
    # everything is okay
    return True, ''


def main():
    # subject ID dialog
    sinfo = {'ID': '', 'Fullscreen': ['Yes', 'No']}
    show_form_dialog(sinfo, validation, order=['ID', 'Fullscreen'])
    sid = int(sinfo['ID'])

    # create logging file
    infoLogger = DataLogger(LOG_FOLDER, str(sid) + '.log', 'info_logger', logging_info=True)
    # create data file
    dataLogger = DataLogger(DATA_FOLDER, str(sid) + '.txt', 'data_logger')
    # save info from the dialog box
    dataLogger.write_json({
        k: str(sinfo[k]) for k in sinfo.keys()
    })
    # create window
    presenter = Presenter(fullscreen=(sinfo['Fullscreen'] == 'Yes'), info_logger='info_logger')

    # load data
    images = presenter.load_all_images(IMG_FOLDER, '.png')
    with open(os.getcwd() + '/questions.pkl', 'rb') as f:
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
    # tracker = eyetracker.EyeTracker(presenter, trackertype='eyetribe')

    # show instructions
    for instr in INSTR_BEGIN:
        presenter.show_instructions(instr)
    # show examples
    for t, img in enumerate(EXAMPLE_QUESTIONS):
        data = show_one_trial(presenter, ex_images[img], black_bg, EXAMPLE_QUESTIONS[img], (resp_feedback, no_resp_feedback))
        infoLogger.logger.info('Writing to data file')
        dataLogger.write_json({'example_trial_index': t, 'response': data})

    # randomization
    imglist = list(questions.keys())
    blocks1 = [k for k in INSTR_BLOCK_THEME.keys() if not k.startswith('fact')]
    blocks2 = [k for k in INSTR_BLOCK_THEME.keys() if k.startswith('fact')]
    random.shuffle(blocks1)
    random.shuffle(blocks2)
    blocks = blocks1 + blocks2  # fact blocks at the end

    # show experiment
    presenter.show_instructions(INSTR_EXP)
    for b in blocks:
        random.shuffle(imglist)
        presenter.show_instructions(INSTR_BLOCKS.format(INSTR_BLOCK_THEME[b]))
        for t, img in enumerate(imglist):
            data = show_one_trial(presenter, images[img], black_bg, questions[img][b], (resp_feedback, no_resp_feedback))
            infoLogger.logger.info('Writing to data file')
            dataLogger.write_json({'block': b, 'trial_index': t, 'response': data})

    # end of experiment
    presenter.show_instructions(INSTR_END)
    infoLogger.logger.info('End of experiment')


if __name__ == '__main__':
    main()
