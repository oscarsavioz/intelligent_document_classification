from PIL import Image


def index_to_class_dict():
	return {
		0: "letter",
		1: "form",
		2: "email",
		3: "handwritten",
		4: "advertisement",
		5: "scientific_report",
		6: "scientific_publication",
		7: "specification",
		8: "file_folder",
		9: "news_article",
		10: "budget",
		11: "invoice",
		12: "presentation",
		13: "questionnaire",
		14: "resume",
		15: "memo"}


def class_to_index_dict():
	return {
		"letter": 0,
		"form": 1,
		"email": 2,
		"handwritten": 3,
		"advertisement": 4,
		"scientific_report": 5,
		"scientific_publication": 6,
		"specification": 7,
		"file_folder": 8,
		"news_article": 9,
		"budget": 10,
		"invoice": 11,
		"presentation": 12,
		"questionnaire": 13,
		"resume": 14,
		"memo": 15
	}


def classes_list():
	return ["letter", "form", "email", "handwritten", "advertisement", "scientific_report",
	        "scientific_publication", "specification", "file_folder", "news_article",
	        "budget", "invoice", "presentation", "questionnaire", "resume", "memo"]


class CustomResizeTransform(object):
	def __init__(self, size, position='center'):
		# Verify if given parameters are correct
		assert isinstance(size, int) or (isinstance(size, tuple) and len(size) == 2)
		assert position in ['top', 'right', 'bottom', 'left', 'center']

		self.size = size
		self.position = position

	def __call__(self, image):
		width, height = image.size

		# Compute resize dimensions
		if isinstance(self.size, int):
			if width > height:
				new_width = self.size
				new_height = int(self.size * height / width)
			else:
				new_width = int(self.size * width / height)
				new_height = self.size
		else:
			new_width, new_height = self.size

		# Resize image with computed dimensions
		image = image.resize((new_width, new_height))

		# Position of resized image
		if self.position == 'top':
			x = 0
			y = 0
		elif self.position == 'right':
			x = new_width - width
			y = 0
		elif self.position == 'bottom':
			x = 0
			y = new_height - height
		elif self.position == 'left':
			x = 0
			y = 0
		else:  # 'center' resize
			x = (new_width - width) // 2
			y = (new_height - height) // 2

		# Create final image with computed dimensions
		new_img = Image.new('RGB', (new_width, new_height))
		new_img.paste(image, (x, y))

		return new_img
