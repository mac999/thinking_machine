# title: thinking machine
# date: 2025.8
# author: taewook kang
# email: laputa99999@gmail.com
# description: media art exhibition 2025. AI x ART.
import pygame, math, sys, time, cv2, numpy as np, threading, random

try:
	import pyttsx3
	from ultralytics import YOLO
	from langchain_community.chat_models import ChatOllama
	from langchain.schema import HumanMessage

except ImportError as e:
	print(f"Missing required library: {e}")
	print("Please install: pip install ultralytics pyttsx3 langchain langchain-community opencv-python")
	exit(1)

# Check camera list
def get_available_cameras():
	"""
	Check and display list of available cameras
	"""
	print("Available camera list:")
	available_cameras = []
	
	# Check up to 10 camera indices
	for index in range(10):
		cap = cv2.VideoCapture(index)
		if cap.isOpened():
			# Get camera information
			width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
			height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
			fps = cap.get(cv2.CAP_PROP_FPS)
			
			available_cameras.append(index)
			print(f"Camera {index}: resolution {int(width)}x{int(height)}, FPS {fps}")
			cap.release()
		else:
			cap.release()
	
	if not available_cameras:
		print("No available cameras found.")
	
	return available_cameras

available_cameras = get_available_cameras()

# 2. Constant definitions
BLACK = (0, 0, 0)
BLUE_TINT = (0, 0, 255, 50)  # Blue color and transparency
RED_TINT = (255, 0, 0, 50)   # Red color and transparency
IMAGE_PATH = 'apple.png'
BEAT_SOUND_PATH = 'heartbeat.mp3'  # Heartbeat sound file
HEARTBEAT_BPM = 40  # 72. Beats per minute
PULSE_DEPTH = 0.15  # Amplitude (up to 15% larger than base size)
PERSON_AREA_THRESHOLD = 0.1  # Person must occupy 10% or more of screen to activate heartbeat

class LLMProcessor:
	"""Local LLM processing using Ollama + Gemma"""
	
	def __init__(self, model_name: str = "tinyllama", base_url: str = "http://localhost:11434"):
		try:
			self.llm = ChatOllama(
				model=model_name,
				base_url=base_url,
				temperature=0.1
			)
			print(f"LLM {model_name} initialized successfully")
		except Exception as e:
			print(f"Failed to initialize LLM: {e}")
			self.llm = None
	
	def process_sentence(self, sentence: str) -> str:
		"""
		Process sentence through LLM
		
		Args:
			sentence: Input sentence to process
		
		Returns:
			Processed sentence from LLM
		"""
		if self.llm is None:
			print("LLM not available, returning original sentence")
			return sentence
		
		try:
			# Simple prompt for now - can be enhanced
			prompt = f"Make a philosophical maxim within 10 words from '{sentence}'"
			message = HumanMessage(content=prompt)
			response = self.llm([message])
			return response.content.strip()
		
		except Exception as e:
			print(f"LLM processing failed: {e}")
			return sentence

LLM_text = ""
llm_thread = None
llm_processor = None
input_sentence = "I see you. Your heart is beating."
last_llm_start_time = 0  # Track last LLM thread start time

def generate_llm_text():
	"""Thread function to run LLM and update global variable with generated text"""
	global LLM_text, llm_processor, input_sentence
	if llm_processor:
		processed_text = llm_processor.process_sentence(input_sentence)
		LLM_text = processed_text
	else:
		LLM_text = ""

class HeartbeatAudio:
	"""Heartbeat audio management class"""
	
	def __init__(self, sound_file: str):
		try:
			# Initialize pygame mixer - adjust settings for MP3 support
			pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=1024)
			pygame.mixer.init()
			
			# Load sound file (MP3 support)
			try:
				self.heartbeat_sound = pygame.mixer.Sound(sound_file)
				# Set sound volume to maximum
				self.heartbeat_sound.set_volume(1.0)
			except pygame.error:
				# If MP3 is not directly supported, use music module
				print(f"Direct MP3 loading failed, trying pygame.mixer.music for '{sound_file}'")
				self.use_music_module = True
				self.sound_file = sound_file
				self.heartbeat_sound = None
			else:
				self.use_music_module = False
			
			# Set overall mixer volume to maximum
			pygame.mixer.music.set_volume(1.0)
			
			self.is_playing = False
			self.sound_channel = None
			
			print(f"Heartbeat sound '{sound_file}' loaded successfully with maximum volume")
			
		except pygame.error as e:
			print(f"Failed to load sound file '{sound_file}': {e}")
			print("Note: Make sure pygame is compiled with MP3 support or convert file to WAV/OGG format")
			self.heartbeat_sound = None
			self.use_music_module = False
	
	def play_heartbeat(self):
		"""Play heartbeat sound if not already playing"""
		if self.use_music_module:
			# Use pygame.mixer.music (for MP3 files)
			if not pygame.mixer.music.get_busy():
				pygame.mixer.music.load(self.sound_file)
				pygame.mixer.music.play()
				self.is_playing = True
		else:
			# Use pygame.mixer.Sound (for WAV/OGG files)
			if self.heartbeat_sound is None:
				return
			
			# Check if currently playing
			if self.sound_channel and self.sound_channel.get_busy():
				self.is_playing = True
				return
			else:
				self.is_playing = False
			
			# Play new sound if not playing
			if not self.is_playing:
				self.sound_channel = self.heartbeat_sound.play()
				self.is_playing = True
	
	def stop_heartbeat(self):
		"""Stop heartbeat sound"""
		if self.use_music_module:
			pygame.mixer.music.stop()
		else:
			if self.sound_channel:
				self.sound_channel.stop()
		self.is_playing = False
	
	def is_sound_playing(self) -> bool:
		"""Check if heartbeat sound is currently playing"""
		if self.use_music_module:
			return pygame.mixer.music.get_busy()
		else:
			if self.sound_channel:
				return self.sound_channel.get_busy()
		return False

class PersonDetector:
	"""YOLO-based person detection for heartbeat trigger"""
	
	def __init__(self, model_name: str = "yolov8n.pt"):
		try:
			self.model = YOLO(model_name)
			print(f"YOLO model {model_name} loaded successfully")
		except Exception as e:
			print(f"Failed to load YOLO model: {e}")
			raise
	
	def detect_person_objects(self, frame: np.ndarray):
		"""
		Detect person in frame and calculate area ratio
		
		Args:
			frame: Input video frame
			
		Returns:
			Ratio of person area to total frame area (0.0 to 1.0), observed_text
		"""
		try:
			results = self.model(frame, verbose=False)
			
			if results[0].boxes is None:
				return 0.0, ''
			
			frame_height, frame_width = frame.shape[:2]
			total_frame_area = frame_width * frame_height
			total_person_area = 0
			count_person = 0
			
			# Dictionary to store class detection info: {class_name: {'count': int, 'confidence_sum': float}}
			class_detections = {}
			
			for box in results[0].boxes:
				class_id = int(box.cls.cpu().numpy()[0])
				confidence = float(box.conf.cpu().numpy()[0])
				class_name = self.model.names.get(class_id, "unknown")
				
				# Count person area for heartbeat trigger
				if class_id == 0 and confidence > 0.6:
					# Get bounding box coordinates
					x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
					person_area = (x2 - x1) * (y2 - y1)
					total_person_area += person_area
					count_person += 1

				# Collect detection statistics for all objects with confidence > 0.2
				if confidence > 0.2:
					if class_name not in class_detections:
						class_detections[class_name] = {'count': 0, 'confidence_sum': 0.0}
					
					class_detections[class_name]['count'] += 1
					class_detections[class_name]['confidence_sum'] += confidence
			
			# Generate observed_text in the requested format
			observed_text = ''
			length = len(class_detections)
			if length > 0:
				observed_text_parts = []
				for _ in range(5):
					random_index = random.randint(0, length - 1)
					class_name, detection_info = list(class_detections.items())[random_index]
					if class_name in observed_text:
						continue

					count = detection_info['count']
					avg_confidence = detection_info['confidence_sum'] / count
					observed_text_parts.append(f"{class_name}({count}: {avg_confidence:.2f})")
					observed_text = ', '.join(observed_text_parts)

			# Calculate ratio of person area to total frame area
			if count_person == 0 or total_frame_area == 0:
				return 0.0, observed_text
			area_ratio = (total_person_area / count_person) / total_frame_area
			return min(area_ratio, 1.0), observed_text  # Cap at 1.0
			
		except Exception as e:
			print(f"Person detection failed: {e}")
			return 0.0, ''

def heartbeat_function(t):
	"""
	Function to simulate heartbeat
	
	Args:
		t: Time in seconds
	
	Returns:
		Heartbeat intensity between 0.0 and 1.0
	"""
	# Convert BPM to beats per second (Hz)
	frequency = HEARTBEAT_BPM / 60.0
	
	# Calculate heartbeat cycle
	cycle_time = t * frequency
	phase = cycle_time - math.floor(cycle_time)  # Cycle position between 0~1
	
	# Simulate actual heartbeat pattern
	if phase < 0.1:
		# First large beat (systole)
		intensity = math.sin(phase * math.pi / 0.1) ** 2
	elif phase < 0.2:
		# Decrease after first beat
		intensity = 0.3 * math.cos((phase - 0.1) * math.pi / 0.1)
	elif phase < 0.35:
		# Second small beat (diastole)
		small_phase = (phase - 0.2) / 0.15
		intensity = 0.4 * math.sin(small_phase * math.pi) ** 1.5
	elif phase < 0.5:
		# Decrease after second beat
		intensity = 0.1 * math.cos((phase - 0.35) * math.pi / 0.15)
	else:
		# Rest period
		rest_phase = (phase - 0.5) / 0.5
		intensity = 0.05 * math.sin(rest_phase * math.pi * 2) * math.exp(-rest_phase * 3)
	
	# Clamp to 0~1 range and ensure minimum value
	return max(0.0, min(1.0, intensity))

# Helper function: apply color tint to image
def create_tinted_surface(surface, tint_color):
	tinted_surface = surface.copy()
	tinted_surface.fill(tint_color, special_flags=pygame.BLEND_RGBA_MULT)
	return tinted_surface

def render_status_text(screen, text, font, color=(255, 255, 255)):
	"""
	Function to render status text at center of screen
	
	Args:
		screen: pygame screen object
		text: Text to render
		font: Font object to use
		color: Text color (default: white)
	"""
	screen.fill(BLACK)  # Clear screen
	text_surface = font.render(text, True, color)
	text_rect = text_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
	screen.blit(text_surface, text_rect)
	pygame.display.flip()

# 3. Pygame initialization
pygame.init()

# Set fullscreen mode (no window frame)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Heartbeat Pulse Effect with Person Detection")
clock = pygame.time.Clock()

# Get actual fullscreen size
SCREEN_WIDTH = screen.get_width()
SCREEN_HEIGHT = screen.get_height()

print(f"Fullscreen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

# 4. Image loading and tinted image creation
try:
	original_apple_img = pygame.image.load(IMAGE_PATH).convert_alpha()
except pygame.error:
	print(f"Cannot find '{IMAGE_PATH}' file.")
	sys.exit()

blue_apple_img = create_tinted_surface(original_apple_img, BLUE_TINT)
red_apple_img = create_tinted_surface(original_apple_img, RED_TINT)

# 5. Initialize YOLO model, camera and audio (after Pygame initialization)
print("Initializing YOLO model...")
person_detector = PersonDetector()

print("Initializing heartbeat audio...")
heartbeat_audio = HeartbeatAudio(BEAT_SOUND_PATH)

llm_processor = LLMProcessor()

font = pygame.font.Font(None, 32) # 32-point font
smallfont = pygame.font.Font(None, 24) # 24-point font (redefined)

render_status_text(screen, "Initializing LLM, YOLO, heartbeat and camera...", smallfont)
print("Initializing LLM, YOLO, heartbeat and camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
	render_status_text(screen, "Camera initialization failed!", smallfont, (255, 0, 0))
	pygame.time.wait(2000)  # Wait 2 seconds
	print("Cannot open camera.")
	print("Possible solutions:")
	print("1. Check if another program is using the camera")
	print("2. Try changing camera index to 1, 2")
	print("3. Reinstall camera driver")
	sys.exit()

# Camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

render_status_text(screen, "All systems initialized. Starting...", smallfont, (0, 255, 0))
pygame.time.wait(1000)  # Wait 1 second
print("Camera initialized successfully")

start_time = time.time()
heartbeat_active = False
last_detection_time = 0

# 6. Main loop
title_text = ''
smallfont = pygame.font.Font(None, 16) # 16-point font (redefined)
running = True
while running:
	# 6-1. Event handling
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		if event.type == pygame.KEYDOWN:
			if event.key == pygame.K_ESCAPE:
				running = False

	# 6-2. Read frame from camera
	ret, frame = cap.read()
	observed_text = ''
	heartbeat_active = False    
	if ret:
		# Detect person and calculate area ratio
		person_area_ratio, observed_text = person_detector.detect_person_objects(frame)
		
		# Activate heartbeat if person occupies more than threshold of screen
		if person_area_ratio >= PERSON_AREA_THRESHOLD:
			heartbeat_active = True
			last_detection_time = time.time()
			print(f"Person detected: {person_area_ratio:.2%} of screen - Heartbeat ON")

			# Start new LLM thread if none exists or previous one finished AND minimum 5 seconds have passed
			current_time = time.time()
			if ((llm_thread is None or not llm_thread.is_alive()) and 
				(current_time - last_llm_start_time >= 5.0)):
				
				input_sentence = observed_text
				title_text = observed_text.split(',')[0] if observed_text else ""

				last_llm_start_time = current_time
				print("Starting LLM text generation thread.")
				llm_thread = threading.Thread(target=generate_llm_text, daemon=True)
				llm_thread.start()

				# Play heartbeat sound (only when not already playing)
				heartbeat_audio.play_heartbeat()

	# 6-3. Animation logic (heartbeat-based scaling)
	elapsed_time = time.time() - start_time

	# 6-4. Rendering
	origin_rect = original_apple_img.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
	screen.fill(BLACK)
	screen.blit(original_apple_img, origin_rect)

	if heartbeat_active:        
		# Calculate pulse intensity using heartbeat function
		pulse_intensity = heartbeat_function(elapsed_time)
		
		# Map scale factor between 1.0 ~ (1.0 + PULSE_DEPTH)
		scale_factor = 1.0 + pulse_intensity * PULSE_DEPTH
		
		scaled_width = int(original_apple_img.get_width() * scale_factor)
		scaled_height = int(original_apple_img.get_height() * scale_factor)

		# Create scaled images
		scaled_blue_img = pygame.transform.smoothscale(blue_apple_img, (scaled_width, scaled_height))
		
		# Red image changes more dramatically with heartbeat intensity
		red_scale_factor = 1.0 + pulse_intensity * PULSE_DEPTH * 1.2
		red_scaled_width = int(original_apple_img.get_width() * red_scale_factor * 0.95)
		red_scaled_height = int(original_apple_img.get_height() * red_scale_factor * 0.95)
		scaled_red_img = pygame.transform.smoothscale(red_apple_img, (red_scaled_width, red_scaled_height))
		
		# Calculate positions for center alignment
		blue_rect = scaled_blue_img.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
		red_rect = scaled_red_img.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))

		screen.blit(scaled_red_img, red_rect)
		screen.blit(scaled_blue_img, blue_rect)
	else:
		# Stop sound when heartbeat is deactivated
		title_text = ''
		LLM_text = ""
		if heartbeat_audio.is_sound_playing():
			heartbeat_audio.stop_heartbeat()

	if len(LLM_text):
		text_surface = font.render(LLM_text, True, (255, 255, 255))
		text_rect = text_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2))
		screen.blit(text_surface, text_rect)

		title_text_surface = font.render(title_text, True, (128, 128, 128))
		title_text_rect = title_text_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 32))
		screen.blit(title_text_surface, title_text_rect)

		text_surface = smallfont.render(observed_text, True, (128, 128, 128))
		text_rect = text_surface.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT - 32))
		screen.blit(text_surface, text_rect)

	pygame.display.flip()
	
	# 6-5. FPS setting
	clock.tick(60)

# 7. Cleanup
heartbeat_audio.stop_heartbeat()
cap.release()
pygame.quit()
sys.exit()