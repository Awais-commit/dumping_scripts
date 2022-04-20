'''
This script is used to change the yolo5 saved inference result to the format given in  mAP library.

This working os this script is to read the text file line by line in the given path.
after reading the line it changes the line i.e.(replace the class 0 with name(MotorBike)) and change the order.
after processing it saves that changing.
for saving the changines it create the new file with the same name in the given path and saves it.
'''
import os
path_to_read = r'/home/risc/Awais/practice/label_correct/labels'
path_to_write = r'/home/risc/Awais/practice/label_correct/changed_labels'

os.chdir(path_to_read)
count = 0
for file in os.listdir():
	if file.endswith('.txt'):
		count+=1
		path_to_read1 = f"{path_to_read}/{file}"
		#print("file num {}: ".format(count) ,path_to_read1)

		with open(path_to_read1) as f:
			for line in f:
				line = line.strip()
				line = line.split(' ')
				#print(line)
				line[0] = "MotorBike"
				line.insert(1, line.pop(-1))
				#print(line)
				line = " ".join(line)
				#print(line)

				path =path_to_read1.split('/')[-1].split('.')[-2]
				with open("/home/risc/Awais/practice/label_correct/changed_labels/00%i.txt" %int(path), "a") as out:
					out.write(line)
					out.write('\n')
					print("Done writing to the file{}".format(path))
					
print("total files processed: ", count)
