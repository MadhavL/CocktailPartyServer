import os
from sphfile import SPHFile


path = './csr_1/'  # Path of root folder containing .sph files

# Replace the file listing code with recursive walk
for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith('.wv1'):
            # Get full path of source .sph file
            sph_path = os.path.join(root, filename)
            # Create output .wav filename with same path structure
            wav_filename = os.path.splitext(filename)[0] + '.wav'
            wav_path = os.path.join(root, wav_filename)
            
            # Convert the file
            try:
                sph = SPHFile(sph_path)
                print(f"Converting: {sph_path} -> {wav_path}")
                sph.write_wav(wav_path, 0, 123.57)  # Customize the period of time to crop
            except Exception as e:
                print(f"Error converting {sph_path}: {str(e)}")



	
	