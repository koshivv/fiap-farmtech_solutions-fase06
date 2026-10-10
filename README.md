# 🐱🚗 FarmTech Solutions — Visão Computacional com YOLOv5

**FIAP · Fase 6 · PBL — Entrega 1**
**Aluno:** Victor Vergara Koshikene — RM 568813

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/koshivv/fiap-farmtech_solutions-fase06/blob/main/VictorVergaraKoshikene_rm568813_pbl_fase6.ipynb)

🎥 **Vídeo de demonstração (YouTube, não listado):** [assistir aqui](COLE_AQUI_O_LINK_DO_VIDEO)

---

## Sobre o projeto

A FarmTech Solutions expandiu seus serviços de IA para além do agronegócio, incluindo saúde animal, segurança patrimonial e controle de acesso. Este projeto simula a visita a um cliente que quer entender, na prática, como funciona um sistema de **visão computacional**.

Treinamos um modelo **YOLOv5** customizado para detectar duas classes bem diferentes entre si:

| Classe | Exemplo de aplicação |
|---|---|
| `gato` | monitoramento de animais em propriedades |
| `carro` | segurança patrimonial e controle de acesso de veículos |

Também comparamos três cenários de treino (**30, 60 e 100 épocas**) para avaliar o impacto do número de épocas em acurácia, erro e desempenho.

> 📓 **Toda a solução está documentada no notebook** [`VictorVergaraKoshikene_rm568813_pbl_fase6.ipynb`](VictorVergaraKoshikene_rm568813_pbl_fase6.ipynb): construção do dataset, rotulação, controle de qualidade, treino, validação, teste e conclusões. Este README é apenas um guia de entrada.

## Como o notebook está organizado

1. Configuração do ambiente (GPU e Google Drive)
2. Construção do dataset a partir do COCO 2017
3. Rotulação no Make Sense AI
4. Controle de qualidade dos rótulos
5. Preparação dos dados para o YOLOv5
6. Treinamento dos cenários de épocas
7. Avaliação e comparação na validação
8. Teste em imagens nunca vistas
9. Conclusões, pontos fortes e limitações

## Dataset em resumo

- **Origem:** imagens do [COCO 2017](https://cocodataset.org) (conjunto `val2017`, que não foi usado no pré-treino do YOLOv5, evitando vazamento de dados).
- **Composição:** 40 imagens de gato e 40 de carro (32 treino / 4 validação / 4 teste cada), mais 5 imagens extras com os dois objetos.
- **Rotulação:** feita manualmente no [Make Sense AI](https://www.makesense.ai), em formato YOLO.

## Destaques dos resultados

O cenário recomendado foi o de **60 épocas**, com o melhor mAP50 na validação e o melhor equilíbrio entre precisão, recall e tempo de treino. A análise completa, incluindo as limitações do experimento, está na Seção 9 do notebook.

![Detecções no conjunto de teste](assets/teste_melhor_cenario.png)

<details>
<summary>Comparação entre os cenários de épocas</summary>

![Métricas por cenário](assets/metricas_comparativas.png)
![Curvas de treino](assets/curvas_comparativas.png)

</details>

## Estrutura do repositório

```
├── VictorVergaraKoshikene_rm568813_pbl_fase6.ipynb   # notebook com a solução completa
├── README.md
├── assets/                                           # imagens usadas neste README
└── dataset/
    ├── manifesto_dataset.csv                         # origem, grupo e divisão de cada imagem
    └── labels/
        ├── treino/                                   # rótulos YOLO do treino (67)
        └── validacao/                                # rótulos YOLO da validação (9)
```

As imagens não estão no repositório. O notebook as baixa diretamente do COCO, com base no manifesto, e copia os rótulos deste repositório quando eles não estão no Google Drive de quem executa.

## Como executar

1. Clique no botão **Abrir no Colab** no topo desta página.
2. Ative a GPU em **Runtime → Change runtime type → T4 GPU**.
3. Execute **Runtime → Run all** e autorize o acesso ao Google Drive quando solicitado.

O notebook cria a pasta `MyDrive/FarmTech_Fase6` no Drive de quem executa, com imagens, rótulos e resultados.

> Como o treino em GPU não é totalmente determinístico, uma nova execução pode gerar métricas ligeiramente diferentes das registradas no notebook.

## Créditos e fontes

- **Dataset:** Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context*. ECCV, 2014. Anotações sob licença CC BY 4.0; imagens do Flickr com licenças individuais, registradas no `manifesto_dataset.csv`.
- **Modelo:** [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5).
- **Rotulação:** [Make Sense AI](https://www.makesense.ai).
