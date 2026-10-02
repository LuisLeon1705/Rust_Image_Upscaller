# Adaptive Upscaler

Este proyecto es un escalador de imagenes y videos adaptativo desarrollado en Rust. Utiliza aceleracion por hardware mediante WebGPU (`wgpu`) para procesar los archivos multimedia de forma rapida y asincrona. Ademas, incluye un servidor web integrado utilizando `axum` para facilitar el acceso a la funcionalidad de escalado y restauracion a traves de una interfaz web o API REST.

## Caracteristicas principales

* **Aceleracion por GPU**: Implementacion de shaders en WGSL para escalado y procesamiento masivo paralelo.
* **Gridding adaptativo (Adaptive Gridding)**: Evaluacion conservadora y auto-expansiva implementada a nivel de shader para procesar las imagenes basado en umbrales de contraste (como se ve en `shader.wgsl`). Opcionalmente puede precomputarse una sola vez por pixel de entrada (`use_precomputed_refinement`, experimental) para acelerar escalas altas a costa de una aproximacion menor en los bordes.
* **Filtros y Restauracion**: Incluye implementaciones de FXAA (Fast Approximate Anti-Aliasing) y tecnicas de restauracion de imagenes (bilateral, mediana, deblocking) integradas directamente en los shaders. El radio espacial del filtro bilateral es configurable por el usuario (`bilateral_radius`).
* **Suavizado geometrico de escalonado (`staircase_fix.rs`)**: Post-proceso en CPU, posterior al escalado por GPU, que corrige el "escalonado" en bordes delgados/alto contraste (lineas de cabello, colmillos, contornos finos) que Lanczos y el gridding no pueden resolver por si solos porque reproducen fielmente el jitter de sub-pixel ya presente en la fuente nativa. Por cada tile pequeno, segmenta el contenido en tantas clases tonales como el contenido realmente tenga (division recursiva por Otsu, sin limite fijo), difumina cada clase por separado y recolorea solo los pixeles cercanos al borde usando el color del vecino original mas cercano (preserva el degradado real en vez de promediarlo). Activado por defecto, con checkbox propio en la UI.
* **Vectorizacion a SVG**: Modo de operacion alterno (`operation_mode=vectorize`, via `vtracer`) que convierte la imagen en un SVG vectorial en vez de escalarla como raster.
* **Procesamiento de video**: Soporte integrado para la lectura, descomposicion en frames y escalado continuo de videos.
* **Servidor web integrado**: Utiliza `axum` y `tower-http` para servir la aplicacion web, manejando peticiones multipart para la carga y descarga de archivos.
* **Control de hilos de CPU**: El pool global de `rayon` se limita a 12 hilos (`CPU_WORKER_THREADS` en `main.rs`) para dejar margen termico en procesos largos, independientemente de cuantos nucleos tenga la maquina.

## Tecnologias utilizadas

* **Rust**: Lenguaje de programacion principal.
* **wgpu**: API de graficos y computo, utilizada para compilar y ejecutar los shaders WGSL.
* **tokio**: Entorno de ejecucion asincrono, util para manejar la concurrencia del servidor.
* **axum**: Framework de enrutamiento y servidor web.
* **image**: Manipulacion, lectura y codificacion de imagenes, con soporte para multiples formatos incluyendo WebP.
* **imageproc**: Umbralizacion de Otsu y desenfoque gaussiano, usados por el suavizado geometrico de escalonado (`staircase_fix.rs`).
* **rayon**: Paralelismo de datos en CPU para la extraccion de tiles, la conversion final f32->u8 y el post-proceso de `staircase_fix.rs`.
* **vtracer**: Vectorizacion raster-a-SVG para el modo de operacion `vectorize`.

## Estructura del proyecto

* `src/`: Contiene el codigo fuente principal en Rust y los shaders.
  * `main.rs`: Punto de entrada del servidor web y configuracion de rutas.
  * `gpu_compute.rs`: Controladores para inicializar e interactuar con el contexto y los buffers de `wgpu`.
  * `upscaler.rs`: Logica principal del proceso de escalado e interaccion con los shaders.
  * `video.rs`: Logica para procesar e iterar flujos de video.
  * `vectorize.rs`: Conversion de raster a SVG mediante `vtracer`.
  * `staircase_fix.rs`: Post-proceso en CPU (tile-based) que suaviza geometricamente el escalonado en bordes finos despues del escalado por GPU — ver el comentario de modulo para el algoritmo completo.
  * `fxaa.wgsl`, `restoration.wgsl`, `shader.wgsl`: Shaders programados en WebGPU Shading Language para llevar a cabo el trabajo grafico pesado.
* `static/`: Archivos estaticos servidos por la aplicacion web.
* `Cargo.toml`: Archivo de declaracion del paquete y sus dependencias.

## Como ejecutar el proyecto

1. Asegurate de tener la herramienta de compilacion de [Rust](https://www.rust-lang.org/) instalada en tu sistema.
2. Clona el repositorio y abre una terminal en la raiz del proyecto.
3. Ejecuta el servidor utilizando Cargo:

```bash
cargo run --release
```

El servidor web se compilara, se iniciara y estara esperando conexiones para procesar las imagenes y videos.

## Testing

La mayoria de los tests son de CPU pura y corren con `cargo test`. Los tests marcados `#[ignore]` en `src/gpu_compute.rs` (modulo `tiling_equivalence_test`) requieren una GPU real y comparan, bit a bit, el resultado de procesar una imagen sin tiling contra procesarla en tiles con corte en seco (sin feathering). **Correlos explicitamente con `cargo test --release -- --ignored` antes de mergear cualquier cambio a `shader.wgsl`, a la logica de padding/tiling en `upscaler.rs`, o al struct `Params` compartido entre shaders** — no hay CI con GPU en este repo que los corra automaticamente.

## Registro de decisiones de arquitectura

**Excepcion puntual al principio de "no tocar los shaders de resampling" (shader.wgsl, restoration.wgsl, fxaa.wgsl):** durante el rediseno del sistema de tiling (eliminacion de feathering a favor de corte en seco con padding determinista), se agregaron los campos `origin_x`/`origin_y` a `Params` y se modifico `shader.wgsl::main()` para calcular la posicion de muestreo (`ix`/`iy`/fraccion) siempre en coordenadas absolutas de la imagen completa, nunca relativas al tile.

Esto **no es un cambio de calidad ni de algoritmo** — Lanczos3, anti-ringing y el adaptive gridding (`contrast_thresh`, `blend_max`, la busqueda de vecindad) quedan exactamente iguales. Es un fix de direccionamiento de coordenadas, de la misma categoria que el fix de overflow de `u32` en `upscaler.rs`: sin el, el resultado de procesar por tiles divergia (no por ruido de redondeo, sino con saltos de color reales) del resultado de procesar la imagen entera, porque la posicion de muestreo se calculaba con aritmetica de punto flotante cuya precision depende de la magnitud absoluta de la coordenada (local al tile vs. global a la imagen) — ver el comentario extenso sobre `Params` en `shader.wgsl` para el mecanismo exacto, confirmado con el test de equivalencia bit-a-bit en `gpu_compute.rs`.

Si en el futuro alguien quiere tocar `shader.wgsl` por una razon de calidad/estetica (no de direccionamiento), este cambio **no es un precedente valido** para saltarse el escrutinio de "no tocar los shaders sin justificacion explicita" — esa regla sigue en pie para cualquier cambio que no sea, como este, una correccion de direccionamiento de coordenadas verificable con el test de equivalencia.

**Diagnostico y fix del "escalonado" en bordes finos (hilos de cabello, colmillos, contornos de alto contraste):** investigacion extensa confirmo que la causa NO es el kernel de resampling (Lanczos vs. bilineal dan el mismo resultado, lo cual descarta "ringing" como causa unica) ni el blend del gridding adaptativo (probado con `blend_max=0` vs `0.5`, resultado identico). La causa real es que cualquier filtro de resampling puntual reproduce fielmente el jitter de sub-pixel que ya existe en la imagen nativa — a mayor escala, ese jitter se magnifica y se percibe como escalon. Un caso particular y mas facil de diagnosticar de este mismo problema: la seccion "Physical Pixel Anchoring" de `shader.wgsl` copiaba el pixel nativo exacto, sin mezcla, en cada posicion de la rejilla de escala — si ese pixel nativo era en si mismo un pixel de transicion/antialiasing (no un color plano), el resultado era un bloque aislado y fuera de contexto ("pixel muerto") al agrandarlo. Fix aplicado: el copiado exacto ahora solo ocurre si la vecindad 3x3 nativa del pixel es plana (bajo contraste); un pixel de borde cae en el Lanczos/gridding normal como sus vecinos — preserva fidelidad exacta donde es inambigua (color plano) sin crear outliers en las transiciones.

La solucion general para el escalonado en si (no solo el caso del anclaje) es `staircase_fix.rs`: en vez de intentar arreglarlo en el dominio del filtro (que no puede, porque el problema no es de filtrado), reconstruye geometricamente un borde suave por tile y recolorea solo los pixeles cercanos a el. Se prototipo primero en Python (trazado de contorno global con OpenCV + KDTree) para validar el enfoque antes de invertir en la version de produccion; esa version Python demostro que el fix de Rust ya converge muy cerca del resultado de un trazado de contorno completo (diferencias de <0.5% de pixeles incluso forzando parametros de suavizado mucho mas agresivos), asi que no se justifico portar el trazado de contorno global (que requeriria componentes conectados en vez de una rejilla fija de tiles).

**Decision pendiente, pospuesta a proposito (no por descuido):** a partir de que factor de escala deja de tener sentido entregar una unica imagen monolitica (PNG/JPEG XL) versus una salida tipo piramide de resolucion / tiles deep-zoom explorable sin cargar el archivo completo. Una salida de 51200x51200 ya es dificil de visualizar con un visor de imagenes normal. No se ha implementado ninguna solucion todavia; queda como pregunta abierta para cuando se aborde la estrategia de salida a gran escala.
