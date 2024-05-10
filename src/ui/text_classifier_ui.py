import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Combobox

from deep_translator import GoogleTranslator

from src.classifiers.text_classifier_complex import TextClassifierComplex
from src.utils.document_parser import DocumentParser


def get_positive_int(entry: tk.Entry):
    try:
        value = int(entry.get())
        if value < 0:
            return 0
        else:
            return value
    except ValueError:
        return 0


class TextClassifierGUI:

    def __init__(self, master):
        self.file_paths = []
        self.results = []
        self.progress = None
        self.train_button = None
        self.save_button = None
        self.load_button = None
        self.__init_ui(master)
        self.__load_model()

    def __load_model(self, path="saved_models/"):
        self.text_classifier = TextClassifierComplex()
        self.text_classifier.load(path)
        self.categories_label.config(text=f'Categories: {self.text_classifier.categories}')

    def __init_ui(self, master):
        self.master = master
        master.title("Text Classifier")
        master.geometry("1000x600")

        self.__show_model_buttons(master)
        self.__show_selection_controls(master)
        self.__show_result_controls(master)

        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        for i in range(2, 5):
            master.grid_columnconfigure(i, weight=1)
        master.grid_rowconfigure(4, weight=1)
        master.grid_rowconfigure(9, weight=1)

    def __show_model_buttons(self, master):
        if self.progress is not None:
            self.progress.destroy()

        self.train_nb_var = tk.IntVar(value=1)
        self.train_nb_checkbox = tk.Checkbutton(master, text="Train NB", variable=self.train_nb_var, font=("Arial", 12))
        self.train_nb_checkbox.grid(row=0, column=0, columnspan=1, pady=5, padx=10, sticky=tk.W)

        self.train_svm_var = tk.IntVar(value=1)
        self.train_svm_checkbox = tk.Checkbutton(master, text="Train SVM", variable=self.train_svm_var,
                                                 font=("Arial", 12))
        self.train_svm_checkbox.grid(row=0, column=1, columnspan=1, pady=5, padx=10, sticky=tk.W)

        self.train_rnn_var = tk.IntVar(value=1)
        self.train_rnn_checkbox = tk.Checkbutton(master, text="Train RNN", variable=self.train_rnn_var,
                                                 font=("Arial", 12))
        self.train_rnn_checkbox.grid(row=0, column=2, columnspan=1, pady=5, padx=10, sticky=tk.W)

        self.train_button = tk.Button(master, text="Train", command=self.train_start, font=("Arial", 12))
        self.train_button.grid(row=0, column=3, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

        self.save_button = tk.Button(master, text="Save", command=self.save_start, font=("Arial", 12))
        self.save_button.grid(row=0, column=4, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

        self.load_button = tk.Button(master, text="Load", command=self.load_start, font=("Arial", 12))
        self.load_button.grid(row=0, column=5, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

    def __show_selection_controls(self, master):
        self.categories_label = tk.Label(master, text=f'Categories:', font=("Arial", 12))
        self.categories_label.grid(row=1, column=0, columnspan=5, sticky=tk.W, padx=10, pady=5)

        self.languages_label = tk.Label(master, text="Select Language:", font=("Arial", 12))
        self.languages_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)

        self.languages_combobox = Combobox(master, values=["English", "Ukrainian"], state="readonly",
                                           font=("Arial", 12))
        self.languages_combobox.current(0)  # Set default language to English
        self.languages_combobox.grid(row=2, column=1, columnspan=2, sticky=tk.W + tk.E, padx=10, pady=5)

        self.select_files_button = tk.Button(master, text="Select Files", command=self.select_files, font=("Arial", 12))
        self.select_files_button.grid(row=5, column=0, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

        self.files_label = tk.Label(master, text="Selected Files:", font=("Arial", 12))
        self.files_label.grid(row=4, column=1, sticky=tk.W, padx=10, pady=5)

        self.files_listbox = tk.Listbox(master, width=40, height=10, font=("Arial", 12))
        self.files_listbox.grid(row=5, column=1, columnspan=5, padx=10, pady=5, sticky=tk.W + tk.E)

    def __show_result_controls(self, master):
        self.detail_var = tk.IntVar(value=1)
        self.detail_checkbox = tk.Checkbutton(master, text="Detailed Results", variable=self.detail_var,
                                              font=("Arial", 12))
        self.detail_checkbox.grid(row=8, column=1, columnspan=1, pady=5, padx=10, sticky=tk.W)

        labels = ["NB Weight:", "SVM Weight:", "RNN Weight:"]
        self.weight_fields = []
        for i in range(3):
            label = tk.Label(master, text=labels[i], font=("Arial", 12))
            label.grid(row=7, column=i + 2, pady=5, padx=10, sticky=tk.W)
            weight_field = tk.Entry(master, font=("Arial", 12), width=10)
            weight_field.delete(0, tk.END)
            weight_field.insert(0, "1")
            weight_field.grid(row=8, column=i + 2, pady=5, padx=10, sticky=tk.W)
            self.weight_fields.append(weight_field)

        self.submit_button = tk.Button(master, text="Submit", command=self.submit, font=("Arial", 12))
        self.submit_button.grid(row=9, column=0, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

        self.copy_button = tk.Button(master, text="Copy Results", command=self.copy_results, state=tk.DISABLED,
                                     font=("Arial", 12))
        self.copy_button.grid(row=10, column=0, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

        self.results_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=10, font=("Arial", 12))
        self.results_text.grid(row=9, column=1, columnspan=5, rowspan=2, padx=10, pady=5, sticky=tk.W + tk.E)

    def __show_progress_bar(self, master):
        if self.train_button is not None:
            self.train_button.destroy()
        if self.save_button is not None:
            self.save_button.destroy()
        if self.load_button is not None:
            self.load_button.destroy()
        self.progress = tk.ttk.Progressbar(master, orient="horizontal", length=200, mode="indeterminate")
        self.progress.start()
        self.progress.grid(row=0, column=0, columnspan=5, pady=10, padx=10, sticky=tk.W + tk.E)

    def select_files(self):
        files = filedialog.askopenfilenames()
        if files:
            self.file_paths = files
            self.files_listbox.delete(0, tk.END)
            for file_path in self.file_paths:
                self.files_listbox.insert(tk.END, file_path)

    def submit(self):
        thread = threading.Thread(target=self.background_submit_processing)
        thread.start()

    def background_submit_processing(self):
        self.__show_progress_bar(self.master)
        language = self.languages_combobox.get().lower()

        texts = [DocumentParser.parse_to_text(file_path) for file_path in self.file_paths]
        if language == "ukrainian":
            texts = [GoogleTranslator(source='uk', target='en').translate(txt) for txt in texts]

        nb_weight = get_positive_int(self.weight_fields[0])
        svm_weight = get_positive_int(self.weight_fields[1])
        rnn_weight = get_positive_int(self.weight_fields[2])

        self.results = [self.text_classifier.predict(text, nb_weight, svm_weight, rnn_weight) for text in texts]

        self.results_text.delete(1.0, tk.END)
        is_detailed = self.detail_var.get() == 1

        for file, category in zip(self.file_paths, self.results):
            if is_detailed:
                self.results_text.insert(tk.END, f"{file} - {category}\n")
            else:
                self.results_text.insert(tk.END, f"{file} - {list(category.keys())[0]}\n")
        self.copy_button.config(state=tk.NORMAL)
        self.__show_model_buttons(self.master)

    def train_start(self):
        thread = threading.Thread(target=self.background_train)
        thread.start()

    def save_start(self):
        thread = threading.Thread(target=self.background_save)
        thread.start()

    def load_start(self):
        thread = threading.Thread(target=self.background_load())
        thread.start()

    def background_train(self):
        self.__show_progress_bar(self.master)
        folder_selected = filedialog.askdirectory()
        use_nb = self.train_nb_var.get() == 1
        use_svm = self.train_svm_var.get() == 1
        use_rnn = self.train_rnn_var.get() == 1
        if folder_selected != "":
            preprocessed_text_df = DocumentParser.parse_and_preprocess_files_to_df(folder_selected)
            accuracy = self.text_classifier.train(preprocessed_text_df=preprocessed_text_df, use_nb=use_nb,
                                                  use_svm=use_svm, use_rnn=use_rnn)
            self.categories_label.config(text=f'Categories: {self.text_classifier.categories}')
            messagebox.showinfo("Training is complete", f'Average accuracy of the model(s): {accuracy}')
        self.weight_fields[0].delete(0, tk.END)
        self.weight_fields[0].insert(0, "0" if self.text_classifier.classifer_nb is None else "1")
        self.weight_fields[1].delete(0, tk.END)
        self.weight_fields[1].insert(0, "0" if self.text_classifier.classifer_svm is None else "1")
        self.weight_fields[2].delete(0, tk.END)
        self.weight_fields[2].insert(0, "0" if self.text_classifier.classifer_rnn is None else "1")
        self.__show_model_buttons(self.master)

    def background_save(self):
        self.__show_progress_bar(self.master)
        folder_selected = filedialog.askdirectory()
        if folder_selected != "":
            self.text_classifier.save(folder_selected)
        self.__show_model_buttons(self.master)

    def background_load(self):
        self.__show_progress_bar(self.master)
        folder_selected = filedialog.askdirectory()
        if folder_selected != "":
            self.__load_model(folder_selected)
        self.weight_fields[0].delete(0, tk.END)
        self.weight_fields[0].insert(0, "0" if self.text_classifier.classifer_nb is None else "1")
        self.weight_fields[1].delete(0, tk.END)
        self.weight_fields[1].insert(0, "0" if self.text_classifier.classifer_svm is None else "1")
        self.weight_fields[2].delete(0, tk.END)
        self.weight_fields[2].insert(0, "0" if self.text_classifier.classifer_rnn is None else "1")
        self.__show_model_buttons(self.master)

    def copy_results(self):
        self.master.clipboard_clear()
        self.master.clipboard_append(self.results_text.get(1.0, tk.END))
        messagebox.showinfo("Copied", "Results copied to clipboard.")


def start():
    root = tk.Tk()
    TextClassifierGUI(root)
    root.mainloop()
