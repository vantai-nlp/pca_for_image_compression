from sklearn.decomposition import PCA
import cv2
import numpy as np

def calculate_ratio_compress(img_shape, n_components):
    return (img_shape[0]*n_components + n_components*img_shape[1]) / (img_shape[0] * img_shape[1]) 

def decompose_matrix(matrix, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    eigen_vectors = pca.components_
    transformed_matrix = np.matmul(matrix, eigen_vectors.T)
    #print(transformed_matrix.shape, eigen_vectors.shape)
    return transformed_matrix, eigen_vectors


class ImageCompress(object):
    def __init__(self, dir_img, n_components) -> None:
        self.dir_img = dir_img
        self.dir_save_img = None
        self.compressed_image = None

    def compress_image(self):
        pass

    def decompress_image(self):
        pass

    def save(self):
        cv2.imwrite(self.dir_save_img, self.compressed_img)
        print('Saved at {}'.format(self.dir_save_img))


class GrayImageCompress(ImageCompress):
    def __init__(self, dir_img, n_components) -> None:
        super().__init__(dir_img, n_components)
        self.img = cv2.imread(self.dir_img, cv2.IMREAD_GRAYSCALE)
        self.shape = self.img.shape
        print('Loaded the gray image', self.shape)
        self.n_components = min(n_components, min(self.shape[0], self.shape[1]))
        self.r = calculate_ratio_compress(self.shape, self.n_components)
        self.dir_save_img = self.dir_img[:-5] + '_gray_compressed.png'

    def compress_image(self):
        img = self.img.copy()
        transformed_gray, eigen_vectors_gray = decompose_matrix(img, self.n_components)
        self.compressed_gray = (transformed_gray, eigen_vectors_gray)
        print('Compressed the image successful')

    def decompress_image(self):
        transformed_gray, eigen_vectors_gray = self.compressed_gray
        # reconstruct image
        reconstructed_gray = np.matmul(transformed_gray, eigen_vectors_gray).astype(np.uint8)
        self.compressed_img = reconstructed_gray
        print('Decompressed the image successful')


class ColorImageCompress(ImageCompress):
    def __init__(self, dir_img, n_components) -> None:
        super().__init__(dir_img, n_components)
        self.dir_save_img = self.dir_img[:-5] + '_compressed.png'
        self.img = cv2.imread(self.dir_img, cv2.IMREAD_COLOR)
        self.shape = self.img.shape
        print('Loaded the color image', self.shape)
        self.n_components = min(n_components, min(self.shape[0], self.shape[1]))
        self.r = calculate_ratio_compress(self.shape, self.n_components)

    def compress_image(self):
        img = self.img.copy()
        B, G, R = cv2.split(img)
        transformed_B, eigen_vectors_B = decompose_matrix(B, self.n_components)
        transformed_G, eigen_vectors_G = decompose_matrix(G, self.n_components)
        transformed_R, eigen_vectors_R = decompose_matrix(R, self.n_components)
        self.compressed_B = (transformed_B, eigen_vectors_B)
        self.compressed_G = (transformed_G, eigen_vectors_G)
        self.compressed_R = (transformed_R, eigen_vectors_R)
        print('Compressed the image successful')

    def decompress_image(self):
        transformed_B, eigen_vectors_B = self.compressed_B
        transformed_G, eigen_vectors_G = self.compressed_G
        transformed_R, eigen_vectors_R = self.compressed_R
        # reconstruct image
        reconstructedB = np.matmul(transformed_B, eigen_vectors_B).astype(np.uint8)
        reconstructedG = np.matmul(transformed_G, eigen_vectors_G).astype(np.uint8)
        reconstructedR = np.matmul(transformed_R, eigen_vectors_R).astype(np.uint8)
        # merge 3 channel image
        self.compressed_img = cv2.merge([reconstructedB, reconstructedG, reconstructedR])
        print('Decompressed the image successful')


if __name__ == '__main__':
    dir_image = './images/the_girl.png'
    compressing_img = ColorImageCompress(dir_image, n_components=200)
    compressing_img.compress_image()
    compressing_img.decompress_image()
    print('r = ',compressing_img.r)
    compressing_img.save()