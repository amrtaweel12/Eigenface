'''
Eng. Amr Nael Zuhdi Taweel, Computer Engineer & Researcher
Date : 10 May 2025
Istanbul Technical University, Computer Engineering Department
EigenFaces Project
'''
import argparse
import numpy as np
from PIL import  Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
class EigenFaces :
    def __init__(self, M,output_path = "",orl_path = "", height =112, width = 92, num_of_images_folder = 10): ##M represent the number of image chose from the dataset
        self.__M = M
        self.__output_path = output_path
        self.__orl_path = orl_path
        if(len(output_path) != 0 and output_path[len(output_path)-1] != "/"):
            self.__output_path = self.__output_path + "/"
        if(len(orl_path) != 0 and orl_path[len(orl_path)-1] != "/"):
            self.__orl_path = self.__orl_path + "/"
        self.__num_of_images_folder = num_of_images_folder
        self.__average_face_vector = None
        self.__images_matrix = None ## this will have the matrix that each row correspond to image vector from database in range of m 
        self.__width = width
        self.__height = height
        self.__image_vector_length = height * width
        self.__eigen_faces_matrix = None ## each row will correspond to eigenface vector
        self.__difference_matrix = None ## it is equal to (images_matrix-average_face_vector) Transpose
        self.__people_classification = None ## this will be used to decide people as classes
        self.__eigen_values_vector = None ## each element will stand for eigenvalue for the covariance matrix and it will be sorted in descending order
        self.__images_to_matrix()
        self.__average_face()
        self.__difference_matrix_calculator()
        self.__covariance_transpose_calculator()
        self.__eigen_values_vectors_calculator()
        self.__covert_image_to_class()
        
    @staticmethod
    def image_to_vector(image_path, width = 92, height = 112, ):
        '''
        This function will take image_path then it will resize the image with specific width and height. convert this image to matrix width * height
          and convert that matrix to vector and return it. If the image path do not exist it will return none'''
        try:
            image = Image.open(image_path).resize((width,height)).convert('L')
            image_matrix = np.array(image, dtype= np.float32)
            image_vector = np.ravel(image_matrix)
            return image_vector
        except(FileNotFoundError,OSError) as e:
            print("Could not open Image with path " + str(image_path))
            return None

    @staticmethod
    def vector_to_matrix(vector , rows, columns):
        ''' This function aims to convert 1d array with length of row * columns to 2d array matrix with demission rows* columns'''
        matrix = np.reshape(vector, (rows, columns))
        return matrix      
    def __images_to_matrix(self):
        '''This function aims to construct matrix of demission M * image_vector_length. This matrix will be constructed by  M images from the available  database and
        make each row represent a vector image '''
        self.__images_matrix = np.zeros((self.__M, self.__image_vector_length))
        for i in range(self.__M):
            image_index = i % self.__num_of_images_folder + 1 ## this will be indexing for the image we are going to choses from file eg 1.pgm
            file_index = int(i / self.__num_of_images_folder) + 1 ## this will be corresponding for the file that will be chose from file set eg s1
            image_path =  self.__orl_path+ "image_set/s"+ str(file_index)+ "/" + str(image_index)+ ".pgm"  ## replace with input path  ##image_path includes the path of the image that we are going to convert to vector
            self.__images_matrix[i] = self.image_to_vector(image_path, self.__width, self.__height)
    def __average_face(self, save = False):
        '''This function is used to calculate the average face  vector depending on the images_matrix that we have'''
        __average_face_vector = np.mean(self.__images_matrix, axis=0).astype(np.float32)
        self.__average_face_vector = __average_face_vector  
    def average_face_image_display(self, new_image_path_name):
        '''This function is used to display the average face vector as an image and save it in a specific path '''
        average_face_matrix  = self.vector_to_matrix(self.__average_face_vector,self.__height, self.__width )
        Image.fromarray(average_face_matrix).convert("L").save( self.__output_path + new_image_path_name)
    def __difference_matrix_calculator(self):
        '''This finds the matrix A which each column in this matrix represent the difference between m image vector and average_face vector'''
        self.__difference_matrix = (self.__images_matrix - self.__average_face_vector).transpose()
    def __covariance_transpose_calculator(self):
        '''The covariance matrix is calculated using AAT which make it dimension hight but if we calculate ATA we can reduce demission M*M and that what this function do'''
        self.__covariance_transpose = self.__difference_matrix.transpose() @ self.__difference_matrix
    def __eigen_values_vectors_calculator(self):
        '''This function calculate the eigen_values and eigen vector for the covariance matrix transpose order them in descending order and normalize the eigenvectors'''
        eigen_values, eigen_vector_conv_trans = np.linalg.eigh(self.__covariance_transpose)
        order = np.argsort(eigen_values)[::-1]
        eigen_values = eigen_values[order]
        self.__eigen_values_vector = eigen_values
        eigen_vector_conv_trans = eigen_vector_conv_trans[:, order]
        eigen_vector = self.__difference_matrix @ eigen_vector_conv_trans #this give us the vector of covariance matrix using covariance transpose 
        eigen_vector = eigen_vector.transpose()
        norms = np.linalg.norm(eigen_vector, axis=1,keepdims= True)
        eigen_vector /= norms
        self.__eigen_faces_matrix = eigen_vector
    def get_eigenvalue(self):
        '''This is get function which returns a copy of eigenvalues array'''
        return self.__eigen_values_vector.copy()
    def show_top_eigen_faces(self,number_of_eigenfaces):
        if number_of_eigenfaces > self.__M:
            number_of_eigenfaces = self.__M
        for i in range(number_of_eigenfaces):
            path = self.__output_path+ "eigenfaces/ef_" + str(i) +"_" +str(self.__M) +".png"
            image_vector_gray = np.round(np.ceil(np.interp(self.__eigen_faces_matrix[i], (self.__eigen_faces_matrix[i].min(), self.__eigen_faces_matrix[i].max()), (0, 255)))).astype(np.uint8)
            image_matrix =   self.vector_to_matrix(image_vector_gray,self.__height, self.__width )
            Image.fromarray(image_matrix).convert("L").save(path)
    def image_reconstruction(self, image_path, image_index,eigenfaces_number,save= True):
        '''image_reconstruction function reconstruct images using specific eigenfaces number. It first project image difference vector into eigenspaces  and find weights then
        using these weights and average face we reconstruct the image'''
        image_vector = self.image_to_vector(self.__orl_path + image_path)
        difference = image_vector - self.__average_face_vector
        weights = np.zeros(eigenfaces_number)
        for i in range(eigenfaces_number): # projection in to eigenface space so we find weights
            weights[i] = np.dot(self.__eigen_faces_matrix[i], difference)
        reconstructed_image =  self.__average_face_vector.copy() ## add average face to the reconstructed image 
        for i in range(eigenfaces_number):
            reconstructed_image += (self.__eigen_faces_matrix[i] * weights[i]) ## reconstructing the image using the weights and eigenfaces 
        error = image_vector - reconstructed_image
        MSE = np.mean(error**2) ##MSE error which is error square / image area
        if(save == True):
            reconstructed_image_matrix = self.vector_to_matrix(reconstructed_image, self.__height ,self.__width)
            Image.fromarray(reconstructed_image_matrix).convert("L").save(self.__output_path + "reconstructed/comparison_" +str(image_index) + "_M" + str(eigenfaces_number)+".png")
        return MSE
    def project_in_eigen_space(self,image_vector, num_of_eigenfaces): ## num og eigenfaces represent how many eigen vector to project on 
        '''This function is used to find the weight of image by projecting them into eigenspace'''
        difference = image_vector - self.__average_face_vector
        weights = self.__eigen_faces_matrix[:num_of_eigenfaces] @ difference
        return weights
    def __covert_image_to_class(self, number_of_people_in_file=10):
        '''This function is used to classification by finding the projection of each person in the eigenspace. This is done by taking a set of images for 
        specific person then project it into eigenspace and take then averaging it'''
        num_people = self.__M // number_of_people_in_file
        diffs = self.__difference_matrix.transpose()
        eigenfaces = self.__eigen_faces_matrix
        all_projections = diffs @ eigenfaces.T
        proj_grouped = all_projections.reshape(
            num_people,
            number_of_people_in_file,
            self.__M
        )
        self.__people_classification = proj_grouped.mean(axis=1)
    def confusion_matrix_calculator(self,number_of_eigenfaces,number_of_people_in_file = 10) :
        '''This function is responsible for calculating confusion_matrix_calculator which is used to check the pyformance of classification'''
        number_of_people = int(self.__M / number_of_people_in_file)
        weight_classes_matrix = self.__people_classification.copy()
        weight_classes_matrix = weight_classes_matrix[:,:number_of_eigenfaces] 
        confusion_matrix = np.zeros(shape=(number_of_people,number_of_people)) 
        for person_index in range(number_of_people):
            for person_images_index in range(number_of_people_in_file):
                image_vector = self.__images_matrix[person_index * number_of_people_in_file + person_images_index]
                image_vector_projection = self.project_in_eigen_space(image_vector,number_of_eigenfaces)
                weight_difference = image_vector_projection - weight_classes_matrix 
                predict = 0
                error =np.dot(weight_difference[0], weight_difference[0])   
                for i in range(weight_difference.shape[0] - 1):
                    new_error = np.dot(weight_difference[i+1],weight_difference[i+1])
                    if(new_error < error): 
                        predict = i + 1
                        error = new_error
                confusion_matrix[predict][person_index] += 1  
        return confusion_matrix, np.trace(confusion_matrix)/self.__M 
    def predict(self,image_path, matrix = None) :
        '''It predicts the person form the photo by projecting it into the eigenspace'''
        weight_classes_matrix = self.__people_classification.copy()
        if image_path !=  "":
            matrix_projection = self.project_in_eigen_space(self.image_to_vector(image_path),50)
        else:
            matrix_projection =   self.project_in_eigen_space(matrix,50)  
        weight_difference = matrix_projection - weight_classes_matrix[:,:50]
        error = np.dot(weight_difference[0], weight_difference[0])
        predict = 1
        for i in range(weight_difference.shape[0] - 1):
                    new_error = np.dot(weight_difference[i+1],weight_difference[i+1])
                    if(new_error < error): 
                        predict = i + 2
                        error = new_error
        return predict
def matrix_to_image(matrix, image_path):
    '''Convert two d matrix into image'''
    lines = ["A = ["]
    for row in matrix:
        formatted_row = [f"{num:6.0f}" for num in row]
        lines.append(" ".join(formatted_row)) 
    lines.append("]")
    image_content = "\n".join(lines)
    font = ImageFont.truetype("DejaVuSansMono.ttf", 24)
    dummy_img = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    bbox = draw.multiline_textbbox((0, 0), image_content, font=font)
    w, h = bbox[2], bbox[3]
    padding = 20
    im = Image.new(mode="RGB", size=(w + 2 * padding, h + 2 * padding), color="white")
    draw = ImageDraw.Draw(im)
    draw.multiline_text((padding, padding), image_content, font=font, fill="black")
    im.save(image_path)
def gaussian_noise(image_matrix, value):
    '''Add gaussian noise to image'''
    image_matrix = image_matrix.astype(np.float32)/255
    row,col = image_matrix.shape
    sigma = value 
    gauss = np.random.normal(0.0,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = np.clip(image_matrix + gauss, 0.0, 1.0)
    noisy = (noisy * 255.0).round().astype(np.uint8)
    return noisy
def salt_pepper_noise(image_matrix,density):
    '''add salt pepper noise into image'''
    row,col = image_matrix.shape
    amount = density
    s_vs_p = 0.5
    out = np.copy(image_matrix)
    num_salt = np.ceil(amount * image_matrix.size * s_vs_p)
    coords = [np.random.randint(0, i, int(num_salt)) for i in image_matrix.shape]
    out[coords[0], coords[1]] = 255
    num_pepper = np.ceil(amount* image_matrix.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i, int(num_pepper)) for i in image_matrix.shape]
    out[coords[0], coords[1]] = 0
    return out
## run it using the following command time python3 eigenfaces.py --output_path output_directory --data_path "" 
##since data path is already in file and nothing between image_set and our eigenfaces.py I make it empty
def main(input_path ,output_path):
    ## task 1
    if(output_path != ""):
        os.makedirs(output_path, exist_ok= True)
    if(len(output_path) != 0 and output_path[len(output_path)- 1 ] != '/'):
        output_path = output_path + '/'
    if(len(input_path) != 0 and input_path[len(input_path)- 1 ] != '/'):
        input_path = input_path + '/'
    set_400_image = EigenFaces(400, output_path, input_path)
    set_10_image = EigenFaces(10,output_path, input_path) 
    set_400_image.average_face_image_display("mean_face_all.png")
    set_10_image.average_face_image_display("mean_face_subset.png")
    ##end of the task 1
    ## task 2
    os.makedirs(output_path + "eigenfaces", exist_ok=True)
    set_10_image.show_top_eigen_faces(10)
    eigen_values_10 = set_10_image.get_eigenvalue()
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], eigen_values_10, label="M-10")
    set_20_imag = EigenFaces(20,output_path, input_path)
    set_20_imag.show_top_eigen_faces(10)
    eigen_values_20 = set_20_imag.get_eigenvalue()
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], eigen_values_20[0:10], label="M-20")
    set_50_imag = EigenFaces(50, output_path, input_path)
    set_50_imag.show_top_eigen_faces(10)
    eigen_values_50 = set_50_imag.get_eigenvalue()
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], eigen_values_50[0:10], label="M-50")
    set_100_imag = EigenFaces(100,output_path, input_path)
    eigen_values_100 = set_100_imag.get_eigenvalue()
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], eigen_values_100[0:10], label="M-100")
    set_100_imag.show_top_eigen_faces(10)
    set_200_imag = EigenFaces(200,output_path, input_path)
    eigen_values_200 = set_200_imag.get_eigenvalue()
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], eigen_values_200[0:10], label="M-200")
    set_200_imag.show_top_eigen_faces(10)
    set_300_imag = EigenFaces(300,output_path, input_path)
    eigen_values_300 = set_300_imag.get_eigenvalue()
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], eigen_values_300[0:10], label="M-300")
    set_300_imag.show_top_eigen_faces(10)
    plt.xlabel("Eigenface Number")
    plt.ylabel("Eigenvalue")
    plt.legend()
    plt.savefig(output_path + "eigenvalues.png")
    plt.close()
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], np.cumsum(eigen_values_10)/np.sum(eigen_values_10), label="M-10")
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (np.cumsum(eigen_values_20)/np.sum(eigen_values_20))[0:10], label="M-20")
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (np.cumsum(eigen_values_50)/np.sum(eigen_values_50))[0:10], label="M-50")
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (np.cumsum(eigen_values_100)/np.sum(eigen_values_100))[0:10], label="M-100")
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (np.cumsum(eigen_values_200)/np.sum(eigen_values_200))[0:10], label="M-200")
    plt.plot([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], (np.cumsum(eigen_values_300)/np.sum(eigen_values_300))[0:10], label="M-300")
    plt.xlabel("Eigenface Number")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.legend()
    plt.savefig(output_path + "cumulative_variance.png")
    ## end of task 2
    #task 3
    os.makedirs(output_path + "reconstructed", exist_ok=True)
    M = [10,20 ,50,100,200,300]
    MSE_file = open(output_path + "reconstructed/mse_reconstruction.txt", "w")
    for i in range(len(M)):
        MSE_file.write("M = " + str(M[i]) + " Image = 10 and MSE =" + str(set_400_image.image_reconstruction(input_path + "image_set/s10/1.pgm",10,M[i])) + "\n")
        MSE_file.write("M = " + str(M[i]) + " Image = 11 and MSE =" + str(set_400_image.image_reconstruction(input_path + "image_set/s11/1.pgm",11,M[i]))+ "\n")
    min,max = 0, 400  ## I created these to variable to use binary search 
    min_m = 0
    while min < max :
        midpoint = int((min + max) / 2)
        if(set_400_image.image_reconstruction(input_path + "image_set/s10/1.pgm",10,midpoint,False) < 500 and set_400_image.image_reconstruction(input_path + "image_set/s11/1.pgm",11,midpoint, False) < 500):
            max = midpoint
        else:
            min = midpoint  + 1
    min_m = min
    MSE_file.write("The minimum m is " + str(min_m))
    ##end of task 3

    ##task 4
    os.makedirs(output_path + "recognition", exist_ok=True)
    accuracy = np.zeros(len(M))
    for i in range(len(M)):
        confusion_matrix, accuracy[i]= set_400_image.confusion_matrix_calculator(M[i])
        matrix_to_image(confusion_matrix, output_path + "recognition/confusion_matrix_ M"+ str(M[i]) + ".png")
    f, ax = plt.subplots()
    ax.plot(M, accuracy, 'o')
    plt.xlabel("Eigen Faces Number")
    plt.ylabel("Recognition Accuracy")
    plt.savefig(output_path + "recognition/accuracy_vs_eigenfaces.png")
    ##end of task4
    ## task 5
    os.makedirs( output_path + "noise", exist_ok=True)
    photos_path = ["1/1.pgm","1/3.pgm", "5/3.pgm","7/3.pgm","8/1.pgm", "10/9.pgm", "18/7.pgm", "19/5.pgm", "25/1.pgm", "30/6.pgm", "32/2.pgm", "35/3.pgm", "39/9.pgm", "40/9.pgm"]
    noise = [0.1,0.2, 0.3, 0.4, 0.5]
    accuracy_noise = np.zeros(5)
    accuracy_salt_pepper =np.zeros(5) 
    person = [1,1,5,7,8,10,18, 19, 25,30, 32,35, 39, 40]
    for i in range (len(photos_path)):
        for x in range(len(noise)):
            new_path = output_path + "noise/noisy_" + str(noise[x]).replace(
                ".",""
            ) +"_"+ str(person[i])+".png"
            Image.fromarray(gaussian_noise(EigenFaces.vector_to_matrix(EigenFaces.image_to_vector(input_path + "image_set/s"+ photos_path[i]),112, 92), noise[x])).convert("L").save(new_path)
            predict = set_400_image.predict(new_path)
            salt_noise = np.ravel(salt_pepper_noise(EigenFaces.vector_to_matrix(EigenFaces.image_to_vector(input_path + "image_set/s"+ photos_path[i]),112, 92),noise[x]))
            predict_salt_pepper = set_400_image.predict("",salt_noise)
            if(predict == person[i]):
                accuracy_noise[x] += 1
            if(predict_salt_pepper == person[i]):
                accuracy_salt_pepper[x]  +=1
    accuracy_noise = accuracy_noise/len(photos_path)
    accuracy_salt_pepper = accuracy_salt_pepper / len(photos_path)
    f, ax = plt.subplots()
    ax.plot(noise, accuracy_noise, label = 'Gaussian Noise')
    ax.plot(noise,accuracy_salt_pepper, label= 'Salt Pepper Noise')
    ax.legend()
    plt.xlabel("Noise")
    plt.ylabel("Recognition Accuracy")
    plt.savefig( output_path + "noise/accuracy_vs _noise.png")
if __name__ == "__main__":      # executed only when script is run directly
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument("--output_path")
    parser.add_argument("--data_path")
    args = parser.parse_args()
    main(args.data_path, args.output_path)