# Advance-Machine-Learning-Project

Langkah 1 - Pilih algoritma machine learning

Pada project ini, saya menggunakan algoritma Ensemble Model. Cara kerja ensemble model dalam memprediksi data tidak hanya menggunakan satu model saja tapi akan menggunakan multiple-model dimana hasil prediksi dari model-model tersebut akan diagregasi yang akan menghasilkan konsesus, yang pada akhirnya kita mendapatkan prediksi yang paling optimal. Hasil prediksi akan menggunakan majority vote, jadi hasil prediksi terbanyak tiap data dalam algoritma akan dipilih sebagai hasil akhir model.

Langkah 2 - Algorithm from scratch

The component of learning

- Hypothesis, set of all splitting variables
  Hyperparameter:
	- number of bootstrap sample (B)
	- number of features (m)

- Parameters
  None, ensemble models is an instance-based/ non parametric algorithm

- Learning Algorithm
  None, we just aggregating each bootstrapped model

Pada decision tree, bagging dan random forest, hyperparameter yang dapat di-tuning anatara lain:

estimator = estimator yang kita pakai e.g. DT, KNN
n_estimators = banyak bootstrap sample yang ingin dibuat
max_features = membuat model di setiap bootstrap sample (nonr, sqrt, log)
random_state = untuk memastikan agar code kita bisa direproduce dengan baik

1. Memperkecil variance yang tinggi atau prediktor yang memiliki kecenderungan overfitting dengan menggunakan mean atau mencari konsensus dari model yang memiliki high-variance
2. Memperkecil variance pada data high-variance yang cenderung overfitting
3. Dibandingkan kita mencari proporsi dari kelas hasil bagging, lebih baik mencari mean dari probability munculnya kelas tersebut yang mana adalah hasil bagging. Jadi kita mencari probabilitas munculnya kelas tersebut, sehingga kita tinggal mencari mean dari probability munculnya kelas tersebut di setiap model bootstrap model.
4. Dengan mencari mean/majority vote dari model kita dengan cara mengagregasi atau mencari konsensus dari setip model yang kita buat ke seluruh bootstrap samplenya, hasilnya bootstrap sample menghasilkan dataset dengan karakteristik tersendiri, menjadi model yang general. Sehingga hal tersebut dapat memperkecil variance.
5. melakukan prediksi dengan menilik prediksi di setiap model-modelnya, keluarkan hasil prediksinya. Terakhir agregasi prediksinya


- pseudocode

create a bootstrapped data
- input: 	
	- n_estimator: number of bootstrapped model
	- n_population: size of training data
	- n_sample: size of bootstrapped samples
- output:	
	- set of bootstrapped sample indices

create a random choose predictors
- input: 	
	- n_estimator: number of bootstrapped model
	- n_population: size of all features available
	- n_sample: size of features to use
- output:	
	- set of bootstrapped features indices


create the ensamble models
- input: 	
	- X: the input data
	- y: the output data
- output:	
	- the model
- hyperparameter
	- B: the number of estimators
	- m: the number of features


Pertama kita membuat base yang dapat kita atur cara mengagregasi atau cara menentukan prediktor apa saja yang dipakai

estimator = estimator yang kita pakai e.g. DT, KNN
n_estimators =  banyak bootstrap sample yang ingin dibuat
max_features = membuat model di setiap bootstrap sample
random_state = untuk memastikan agar code kita bisa direproduce dengan baik

1. membuat iterasi pembuatan model bagging/random forest
- Harus memiliki list data model yang ingin dimodelkan untuk memilih data di fitur mana yang ingin kita bootstrap-kan
- modelkan data bootstrap, data diobservasi mana yang akan kita bootstrap-kan
- kita akan membuat model di setiap sample bootstrap kita

2. Lalu kita akan membuat fungsi untuk memproses Random Forest, fungsi tersebut berfungsi untuk bisa memfilter berapa banyak fitur dan fitur apa saja yang kita akan pakai

konsepnya adalah random forest akan melihat kondisi dimana sebuah fitur tidak dipakai, kemudian akan dibandingkan dengan kondisi fitur lain yang tidak dipakai, kemudian dibandingkan performanya.

3. Membuat fungsi predict
- revisit di setiap model-modelnya, keluarkan hasil prediksinya
- agregasi prediksinya

_generate_ensemble_estimators : menunjukan banyaknya estimator pada objek tersebut
_generate_random_seed : untuk mengenerate randomize index
_generate_sample_indices : untuk mengenerate sample indices, kita ingin menyimpan index dari sample masing-masing data
_generate_feature_indices : fungsi untuk memilih feature index, berapa banyak jumlah fitur yang akan kita pilih


langkah 3 - Kesimpulan, analisa, & referensi

- Analisa

-- Bootstrap
Salah satu model yang kita gunakan untuk ensemble model adalah Bootstrap. Ide dasar dari model tersebut adalah melakukan pengambilan data sejumlah data kita (n) dan proses pengambilan data dilakukan dengan pengembalian. Jadi jika kita lakukan proses tersebut secara berulang-ulang, kita akan mendapatkan true distribution dari data yang kita modelkan. Dengan metode Bootstrap ini kita bisa membuat distribusi empiris sehingga kita bisa mengetahui estimasi populasi yang tidak kita ketahui sebelumnya. 
Prosesnya:
- melakukan pengambilan data sample sejumlah n, dimana setiap pengambilan sample dilakukan secara uniform, setiap data memiliki peluang yang sama untuk terambil dan juga pengambilan ini dilakukan dengan pegembalian.
- kita mengetahui bahwa data kita memiliki jumlah X dan Y yang begitu banyak, untuk itu kita hanya akan melakukan resampling terhadap baris datanya.
- proses tersebut dapat menggeneralisasi populasi, sehingga jika dilakukan agregasi, kita akan mendapatkan distribusi empiris yang digunakan untuk mengestimasi distribusi asli dari populasi data.

-- Bagging
Decision Tree cenderung menghasilkan variance yang begitu tinggi, jadi perubahan kecil dari data akan menimbulkan efek yang besar terhadap hasil prediksi, jadi kita akan menggunakan metode Bagging (Bootstrap Aggregating). Ide dasarnya adalah kita akan memperkecil variance yang tinggi atau prediktor yang memiliki kecenderungan overfitting dengan menggunakan mean atau mencari konsensus dari model yang memiliki high-variance tadi. 
Proses:
- pertama kita buat sample bootsrap(B), nilai tersebut dapat kita atur sesuai kebutuhan. Nilai B memiliki ukuran yang sama dengan n
- kedua kita membuat base model(fitting) untuk setiap bootsrap yang kita buat, model yang kita gunakan adalah model yang memiliki nilai variance yang tinggi
- kita mencari cara untuk menurunkan variance yang tinggi dengan cara melakukan agregasi/ mencari konsensus dari setiap model yang kita buat ke seluruh bootstrap sample-nya.
- Hasilnya, bootstrap sample memberikan dataset dengan karakteristik tersendiri, menjadi model yang general untuk memprediksi seluruh karakter secara general.

Dibandingkan kita mencari proporsi dari kelas hasil bagging, lebih baik mencari mean dari probability munculnya kelas tersebut yang mana adalah hasil bagging. Jadi kita mencari probabilitas munculnya kelas tersebut, sehingga kita tinggal mencari mean dari probability munculnya kelas tersebut di setiap model bootstrap model.

validasi pada bagging
- rata-rata kita menggunakan 2/3 data untuk melakukan observasi/training, sisanya kita sebut degnan out of bag observation. Data OOB ini bersifat independen atau tidak memiliki korelasi, untuk itu bisa kita gunakan untuk mengestimasi error, melakukan evaluasi dan validasi.
- caranya, kita melakukan prediksi respon pada data ke-i menggunakan setiap model tree yang ada dalam model Bootstrap, lalu kita melakukan proses agregasi pada setiap prediksi yang ada, kemudian bisa kita temukan error yang kemudian kita cari mean dari errornya.
- kelemahannya adalah kita tidak bisa merepresentasikan diagonal decision rule, kita akan kehilangan interpretabilitas karena hasil bagging bukanlah tree dan yang terakhir adalah perhitungan semakin kompleks karena kita mengalikan pertumbuhan setiap pohon sebanyak B.

-- Random forest
Model yang mirip dengan bagging, yaitu menggunakan bootstrap dan juga membangun sub-sub model untuk setiap bootstrap sample yang sudah dibuat. Bedanya, Random forest dapat lebih memberikan imporvement dengan cara membuat tree yang tidak berkorelasi antara satu pohon dengan pohon lainnya, dengan begitu metode ini dapat lebih meminimalisir high-variance.
Pohon-pohon tersebut akan dilatih berdasarkan pada bootstrap yang kita hasilkan. Namun di setiap proses trainingnya, masing-masing decision tree akan memilih subset dari seluruh prediktor yang ada, agar antar satu pohon dengan pohon yang lain tidak memiliki korelasi. Dalam metode ini, kita juga akan menambahkan randomness, dimana hal tersebut akan membuat pohon satu dengan pohon lain lebih tidak berkorelasi.

Proses:
- Jika kita memiliki dataset berukuran n, kita akan membuat bootstrap sebanyak B (z*)
- Lalu kita bangun tree yang spesifik untuk memprediksi bootstrap sample, dalam proses ini kita mengambil m fitur dari p fitur secara random. Jadi perbedaannya dengan bagging adalah pada pemilihan fitur yang akan diprediksi untuk proses pembuatan model
- Random forest memiliki error lebih kecil dibanding bagging, dikarenakan random forest menghasilkan model yang variance-nya lebih kecil

Yang bisa kita atur adalah jumlah bootstrap sample yang akam kita pilih, jika kita memilih jumlah yang kecil maka errornya akan fluktuatif, namun jika kita perbesar jumlah pohon, maka kita akan mendapatkan performa yang lebih konvergen.

- kesimpulan

random forest & bagging memiliki persamaan dalam input data yaitu membutuhkan seberapa banyak bootstrap sample yang dibuat dan seberapa banyak prediktor yang dipakai untuk memodelkan masing-masing bootstrap sample cara agregasi juga sama, yaitu mencari mean atau majority vote untuk melakukan prediksi. Pembeda dari kedua method tersebut adalah jika bagging menggunakan semua fitur, random forest hanya menggunakan beberapa feature, dan juga random forest menambahkan unsur randomness sehingga membuat data lebih tidak terkolerasi

Kita bisa melihat perbedaan dari decision tree, bagging tree dan random forest, deicision tree memiliki variance yang tinggi seperti kita tahu bahwa training data pada DT bisa mencapai 100 persen. Untuk bagging tree, kita mencoba untuk memperkecil variance dengan mencari mean/majority vote dari model kita dengan cara mengagregasi atau mencari konsensus dari setip model yang kita buat ke seluruh bootstrap samplenya, hasilnya bootstrap sample menghasilkan dataset dengan karakteristik tersendiri, menjadi model yang general. Random forest memiliki input data yang mirip dengan bagging, namun yang jadi pembeda adalah jika bagging menggunakan semua fitur, random forest hanya menggunakan beberapa feature, dan juga random forest menambahkan unsur randomness sehingga membuat data lebih tidak terkolerasi

Hasilnya model semakin kompleks setelah diperbanyak, tapi uniknya setelah model diperbanyak membuat variance semakin turun sehingga model semakin konverse, performance akan semakin stabil pada akhirnya jika terdapat perubahan minor pada data tidak memiliki pengaruh terhadap hasil prediksi. Hal tersebut adalah jawaban dari permasalahan awal kita tadi.

Yang menjadikan sebuah model signifikan adalah selisih antara akurasi training dan akurasi validation yang kecil, jika selisih antara keduanya kecil maka kita semakin yakin bahwa model kita tidak menghafal namun belajar



reference:
- https://arxiv.org/pdf/1106.0257.pdf




