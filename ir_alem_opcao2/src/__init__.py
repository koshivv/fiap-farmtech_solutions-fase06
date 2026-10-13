"""Código reutilizável do "Ir Além — Opção 2" (Transfer Learning, Fine Tuning e Segmentação).

Módulos:
    config        constantes do experimento (sementes, tamanhos, hiperparâmetros, caminhos)
    ambiente      detecção de Colab/local e escolha do backend do Keras (antes de importar o keras)
    dados         manifesto, anotações COCO, download, teste ampliado, carregamento de imagens
    aumento       augmentation da Entrega 2 com as mesmas operações PIL do torchvision
    segmentacao   segmentadores (YOLO-seg / SAM), política de falha, máscaras GT e IoU
    modelos       CNN do zero (port Keras da Entrega 2) e EfficientNetB0 (Transfer Learning)
    treino        laços de treino com checkpoint pela menor perda de validação
    metricas      métricas por classe, testes estatísticos e latência
    visualizacao  grades, curvas, matrizes de confusão, CAM e figura do pipeline
"""
