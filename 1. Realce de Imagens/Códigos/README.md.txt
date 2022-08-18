Para executar qualquer um dos códigos de teste utilize o seguinte comando

>python test_name.py

Esses testes fazem chamadas para funções de bib_pdi.py que contém

- filterMean()
- filterMedian()
- filterGaussian()
- filterLaplacian()
- filterPrewitt()
- filterSobel()
 
Todas essas utilizam a chamada do filter2D(img, kernel)

Nessa biblioteca também tem as funções

- calcHistogram()
- equalizeHistogram()

Testes:

test_average.py : Aplica o kernel da média de tamanhos 3, 5, 9, 15 e 23
test_median.py: Aplica o kernel da mediana de tamanhos 3, 5, 9, 15 e 23
test_gaussian.py: Aplica o kernel gaussiano de tamanhos 3, 5, 9, 15 e 23

test_average.py : Aplica o kernel da media 10x e 20x.
test_median.py: Aplica o kernel da mediana 10x e 20x.

test_laplacian.py: Aplica o filtro laplaciano não estendido e o estendido.
test_sobel.py: Aplica o operador de Sobel.
test_prewitt.py: Aplica o operador de Prewitt.

test_equalize: Aplica a equalização em uma imagem.