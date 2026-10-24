# Prototipo: recomendar vendedores según riesgo de retraso

Es el tercer uso de la misma señal de riesgo de la propuesta del 7 de octubre ("una señal, tres usos"): logística vigila los pedidos de riesgo alto (hecho), quien define las promesas ajusta los días por ruta (propuesto) y, antes de comprar, el cliente ve los vendedores con menos retraso al mismo precio (este prototipo). **Es un prototipo de visión, no parte del producto:** no usa el modelo ni cambia la app o la API.

## Por qué es un prototipo y no parte de la app

- **Es otro usuario y otro momento.** La app sirve a operaciones después de aprobar el pago: qué pedidos vigilar. Este prototipo sirve al cliente antes de comprar: a qué vendedor comprarle.
- **No usa el modelo.** Ordena a los vendedores por su tasa de retraso histórica, no por la predicción calibrada. No pasó por la validación temporal, la calibración ni la elección de modelos que respaldan la app.
- **La evidencia es observacional.** Que los vendedores de menor riesgo se retrasen menos no prueba que recomendarlos reduzca el retraso. Para eso hace falta una prueba A/B.
- **Los datos no alcanzan para un producto.** Solo vemos lo que se vendió, no qué vendedores ofrecían cada producto. Solo el 63 % de los pedidos de 2018 tiene un vendedor con historial, y 7 de cada 10 retrasos ocurren en el transporte (ver "Límites").
- **Está fuera del alcance acordado.** El README deja la recomendación fuera del alcance del proyecto; integrarla sería un cambio de alcance que decide el equipo.
- **No arriesga el producto.** Al estar aparte, si algo falla aquí, la app, la API y sus tests no se ven afectados.

Por eso vive en `prototipos/`: para mostrar la visión de "una señal, tres usos" y los datos que harían falta para construirla de verdad.

## Cómo correrlo

Desde la raíz del repo, con los 9 CSV en `data/raw/`:

```bash
streamlit run prototipos/recomendador_vendedores.py
python -m pytest tests/test_recomendador.py
```

## Cómo ordena a los vendedores

`src/recomendador.py`:

1. **Pedidos de un solo vendedor** (94.931 de los 96.203 con target), para atribuir el retraso sin ambigüedad. El target es el del proyecto (`build_target`, decisión 2).
2. **Historial antes del corte** (compras de 2017): tasa de retraso de cada vendedor con al menos 10 pedidos (641 vendedores).
3. **Puntaje suavizado:** `(retrasos + 20 × tasa media) / (pedidos + 20)`, con una tasa media de 5,7 %. Sin suavizar, un vendedor con 0 % en 10 pedidos quedaría primero por suerte.
4. **Medición después del corte** (compras de 2018), con pedidos que no se usaron para ordenar. Un test comprueba que cambiar lo que pasó en 2018 no mueve el puntaje.

## Resultado

Pedidos de 2018, por tercio del puntaje del vendedor:

| Tercio | Retraso en 2018 | Ajustado por estado del cliente | Precio mediano |
|---|---|---|---|
| Menor riesgo | 6,7 % | 6,7 % | R$ 69,90 |
| Medio | 8,0 % | 8,1 % | R$ 78,00 |
| Mayor riesgo | 9,9 % | 9,8 % | R$ 69,99 |

El orden de 2017 se sostiene en 2018, no se explica por la geografía del cliente y los vendedores de menor riesgo no son más caros. Con un mínimo de 5 o 20 pedidos, o sin suavizar, da casi lo mismo (6,6–6,9 % contra 9,8–9,9 %).

La propuesta del 7 de octubre decía 6,4 % contra 8,3 %, con un método que no quedó documentado. Las cifras de este documento son las que reproduce el código.

## Límites

- **El vendedor explica solo una parte del retraso.** De los pedidos que llegaron tarde, solo el 27,7 % tuvo un vendedor que entregó a la transportadora después de su plazo límite (`shipping_limit_date`). En los otros 7 de cada 10, el retraso ocurrió en el transporte. Cuando el vendedor despacha tarde, el pedido llega tarde el 21,1 % de las veces, contra 5,4 % cuando despacha a tiempo. La fecha de despacho se usa solo para este diagnóstico: es posterior a la compra y nunca entra al modelo (decisión 4).
- **Funciona en promedio, no vendedor por vendedor.** Un vendedor con 0 % en 2017 puede tener más de 25 % en 2018.
- **Cobertura parcial:** 63 % de los pedidos de 2018 tiene un vendedor con historial. Los nuevos no se pueden ordenar.
- **Solo vemos lo vendido,** no qué vendedores ofrecían el mismo producto, a qué precio ni con qué stock.
- **Es observacional.** Los clientes no eligieron vendedor al azar; ajustar por el estado del cliente no cambia el resultado, pero puede haber otras diferencias (tipo de producto, ruta). El efecto de recomendar se mediría con una prueba A/B.
- **Datos de 2017–2018.**

## Qué datos necesitaríamos de Olist

| Dato | Para qué |
|---|---|
| Transportadora y modalidad de envío de cada pedido | Separar el retraso del vendedor del retraso del transporte |
| Eventos de seguimiento con fecha (recolección, centros de distribución, intento de entrega) | Saber en qué tramo de la ruta se pierde el tiempo |
| Catálogo de ofertas: qué vendedores ofrecían cada producto, con precio y stock | Recomendar entre alternativas reales |
| Ubicación real del depósito del vendedor | Calcular la distancia desde donde sale el paquete |
| Tiempo de preparación declarado y capacidad del vendedor | Anticipar demoras de despacho en picos |
| Qué vendedores vio el cliente y cuál eligió | Medir el efecto de recomendar con una prueba A/B |
| Pedidos posteriores a 2018, con reclamos y cancelaciones | Validar que el orden sigue vigente e incluir pedidos que nunca llegaron |

El primer dato importa también para el modelo principal: si 7 de cada 10 retrasos ocurren en el transporte, la transportadora es la variable que más falta.
