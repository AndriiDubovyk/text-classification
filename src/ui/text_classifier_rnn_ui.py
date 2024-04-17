import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Combobox

from deep_translator import GoogleTranslator

from src.models.document_parser import DocumentParser
from src.models.text_classifier_rnn import TextClassifier


class TextClassifierGUI:
    def __load_model(self, path="models/rnn/"):
        self.text_classifier = TextClassifier()
        self.text_classifier.load(path)
        self.categories_label.config(text=f'Categories: {self.text_classifier.categories}')

    def __init_ui(self, master):
        self.master = master
        master.title("Text Classifier")
        master.geometry("800x600")

        self.__show_model_buttons(master)

        self.categories_label = tk.Label(master, text=f'Categories:',
                                         font=("Arial", 12))
        self.categories_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, padx=10, pady=5)

        self.languages_label = tk.Label(master, text="Select Language:", font=("Arial", 12))
        self.languages_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)

        self.languages_combobox = Combobox(master, values=["English", "Ukrainian"], state="readonly",
                                           font=("Arial", 12))
        self.languages_combobox.current(0)  # Set default language to English
        self.languages_combobox.grid(row=2, column=1, columnspan=2, sticky=tk.W + tk.E, padx=10, pady=5)

        self.select_files_button = tk.Button(master, text="Select Files", command=self.select_files, font=("Arial", 12))
        self.select_files_button.grid(row=3, column=0, columnspan=3, pady=10, padx=10, sticky=tk.W + tk.E)

        self.files_label = tk.Label(master, text="Selected Files:", font=("Arial", 12))
        self.files_label.grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)

        self.files_listbox = tk.Listbox(master, width=40, height=10, font=("Arial", 12))
        self.files_listbox.grid(row=5, column=0, columnspan=3, padx=10, pady=5, sticky=tk.W + tk.E)

        self.detail_var = tk.IntVar(value=1)
        self.detail_checkbox = tk.Checkbutton(master, text="Detailed Results", variable=self.detail_var,
                                              font=("Arial", 12))
        self.detail_checkbox.grid(row=6, column=0, columnspan=2, pady=5, padx=10, sticky=tk.W)

        self.submit_button = tk.Button(master, text="Submit", command=self.submit, font=("Arial", 12))
        self.submit_button.grid(row=6, column=1, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

        self.copy_button = tk.Button(master, text="Copy Results", command=self.copy_results, state=tk.DISABLED,
                                     font=("Arial", 12))
        self.copy_button.grid(row=7, column=0, columnspan=3, pady=10, padx=10, sticky=tk.W + tk.E)

        self.results_text = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=40, height=10, font=("Arial", 12))
        self.results_text.grid(row=8, column=0, columnspan=3, padx=10, pady=5, sticky=tk.W + tk.E)

        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=1)
        master.grid_rowconfigure(4, weight=1)
        master.grid_rowconfigure(7, weight=1)

    def __show_model_buttons(self, master):
        if self.progress is not None:
            self.progress.destroy()

        self.train_button = tk.Button(master, text="Train", command=self.train_start, font=("Arial", 12))
        self.train_button.grid(row=0, column=0, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

        self.save_button = tk.Button(master, text="Save", command=self.save_start, font=("Arial", 12))
        self.save_button.grid(row=0, column=1, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

        self.load_button = tk.Button(master, text="Load", command=self.load_start, font=("Arial", 12))
        self.load_button.grid(row=0, column=2, columnspan=1, pady=10, padx=10, sticky=tk.W + tk.E)

    def __show_progress_bar(self, master):
        if self.train_button is not None:
            self.train_button.destroy()
        if self.save_button is not None:
            self.save_button.destroy()
        if self.load_button is not None:
            self.load_button.destroy()
        self.progress = tk.ttk.Progressbar(master, orient="horizontal", length=200, mode="indeterminate")
        self.progress.start()
        self.progress.grid(row=0, column=0, columnspan=3, pady=10, padx=10, sticky=tk.W + tk.E)

    def __init__(self, master):
        self.file_paths = []
        self.progress = None
        self.train_button = None
        self.save_button = None
        self.load_button = None
        self.__init_ui(master)
        self.__load_model()

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

        predicted_categories = [self.text_classifier.predict(text) for text in texts]

        self.results_text.delete(1.0, tk.END)
        is_detailed = self.detail_var.get() == 1

        for file, category in zip(self.file_paths, predicted_categories):
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
        if folder_selected != "":
            accuracy = self.text_classifier.train(folder_selected)
            self.categories_label.config(text=f'Categories: {self.text_classifier.categories}')
            messagebox.showinfo("Training is complete", f'Accuracy of the model: {accuracy}')
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
        self.__show_model_buttons(self.master)

    def copy_results(self):
        self.master.clipboard_clear()
        self.master.clipboard_append(self.results_text.get(1.0, tk.END))
        messagebox.showinfo("Copied", "Results copied to clipboard.")


def start():
    root = tk.Tk()
    TextClassifierGUI(root)
    root.mainloop()
