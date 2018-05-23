class DocumentExample:
    def __init__(self, unique_id, label, text):
        self.unique_id = unique_id
        self.label = label
        self.text = text
        self.tokens = None
        self.dependencies = None
        self.lda_embedding = None
