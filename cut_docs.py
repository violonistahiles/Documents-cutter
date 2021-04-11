import numpy as np
import PySimpleGUI as sg
import os
from time import gmtime, strftime
import imageio
import shutil
import cv2
from PIL import Image, ImageTk
import onnxruntime
from utils import *


def start_layout(images_path):
    images_number = len(os.listdir(images_path))

    layout = [[sg.Text('Select document images')],
              [sg.InputText('Images folder', key='images'), sg.FileBrowse()],
              [sg.Text('Place for document filename', key='filename', size=(50, 1))],
              [sg.Image(key="original_image"), sg.Image(key="predicted_image")],
              [sg.Column([[sg.Button(button_text='Cut', key='cut'),
                           sg.Button(button_text='Clear memory', key='clear_memory'),
                           sg.Text(f'{images_number} images in memory', key='results', size=(25, 1))]]),
               sg.Column([[sg.Cancel('Exit')]], element_justification='right', expand_x=True)],
              [sg.InputText('PDF_filename', key='pdf_filename', size=(25, 1)),
               sg.Button(button_text='Save PDF', key='save_pdf')]
              ]

    return layout


def prepare_image(image):
    # image = Image.open(path)
    width, height = image.size
    scale = height / 300
    new_w, new_h = int(width / scale), int(height / scale)
    # print(new_w, new_h)
    image.thumbnail((new_w, new_h))
    photo_img = ImageTk.PhotoImage(image)

    return photo_img


if __name__ == '__main__':

    sg.theme('DarkAmber')  # Add a little color to your windows

    # Set directories
    module_dir = os.path.dirname(os.path.abspath(__file__))
    images_path = os.path.join(module_dir, 'images')
    model_path = os.path.join(module_dir, 'onnx_model.onnx')
    model = onnxruntime.InferenceSession(model_path)
    results_dir = os.path.join(module_dir, 'results')
    document_height = 1500

    # Generate start window
    layout = start_layout(images_path)
    window = sg.Window('Documents cutter', layout)

    while True:
        event, values = window.read()

        # Case of "Exit" button
        if event in (sg.WIN_CLOSED, 'Exit'):
            break

        # Case of "Cut" button
        if event == 'cut':
            file_path = values['images']
            if os.path.exists(file_path):
                images_number = len(os.listdir(images_path))
                doc_filename = file_path[file_path.rindex('/')+1:file_path.rindex('.')]

                # Cut document from chosen image
                pred_path = os.path.join(images_path, f'{images_number + 1}.png')
                original_image_array = predict(file_path, pred_path, model)
                predicted_image = Image.open(pred_path)
                original_image = Image.fromarray(original_image_array)

                # Update application interface information
                original_image_photo = prepare_image(original_image)
                predicted_image_photo = prepare_image(predicted_image)
                window['results'].Update(f'{images_number + 1} images in memory')
                window['filename'].Update(doc_filename)
                window['original_image'].update(data=original_image_photo)
                window['predicted_image'].update(data=predicted_image_photo)

        # Case of "Clear memory" button
        if event == 'clear_memory':
            if os.path.exists(images_path) is True:
                shutil.rmtree(images_path)
            os.mkdir(images_path)
            window['results'].Update(f'0 images in memory')

        # Case of "Save PDF" button
        if event == 'save_pdf':
            images_for_pdf = []
            # Load and reshape images
            for i, filename in enumerate(sorted(os.listdir(images_path))):
                img = Image.open(os.path.join(images_path, filename))
                img = img.convert('RGB')
                scale = img.height / document_height
                new_w, new_h = int(img.width / scale), int(img.height / scale)
                img = img.resize((new_w, new_h))
                images_for_pdf.append(img.copy())

            # Generate .pdf filename
            pdf_filename = values['pdf_filename']
            if pdf_filename == 'PDF_filename':
                str_time = strftime('%d%m_%H%M%S', gmtime())
                pdf_path = os.path.join(results_dir, f'{str_time}.pdf')
            else:
                pdf_path = os.path.join(results_dir, pdf_filename + '.pdf')

            # Save .pdf
            if len(images_for_pdf) > 1:
                images_for_pdf[0].save(pdf_path, save_all=True, append_images=images_for_pdf[1:])
            elif len(images_for_pdf) == 1:
                images_for_pdf[0].save(pdf_path)

    window.close()
