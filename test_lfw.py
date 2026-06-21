from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=2, resize=1.0)
print(lfw_people.images.dtype, lfw_people.images.shape)
print(lfw_people.images.max())
