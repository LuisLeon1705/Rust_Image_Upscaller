# Adaptive Upscaler

Este proyecto es un escalador de imagenes y videos adaptativo desarrollado en Rust. Utiliza aceleracion por hardware mediante WebGPU (`wgpu`) para procesar los archivos multimedia de forma rapida y asincrona. Ademas, incluye un servidor web integrado utilizando `axum` para facilitar el acceso a la funcionalidad de escalado y restauracion a traves de una interfaz web o API REST.

## Caracteristicas principales

* **Aceleracion por GPU**: Implementacion de shaders en WGSL para escalado y procesamiento masivo paralelo.
* **Gridding adaptativo (Adaptive Gridding)**: Evaluacion conservadora y auto-expansiva implementada a nivel de shader para procesar las imagenes basado en umbrales de contraste (como se ve en `shader.wgsl`).
* **Filtros y Restauracion**: Incluye implementaciones de FXAA (Fast Approximate Anti-Aliasing) y tecnicas de restauracion de imagenes integradas directamente en los shaders.
* **Procesamiento de video**: Soporte integrado para la lectura, descomposicion en frames y escalado continuo de videos.
* **Servidor web integrado**: Utiliza `axum` y `tower-http` para servir la aplicacion web, manejando peticiones multipart para la carga y descarga de archivos.

## Tecnologias utilizadas

* **Rust**: Lenguaje de programacion principal.
* **wgpu**: API de graficos y computo, utilizada para compilar y ejecutar los shaders WGSL.
* **tokio**: Entorno de ejecucion asincrono, util para manejar la concurrencia del servidor.
* **axum**: Framework de enrutamiento y servidor web.
* **image**: Manipulacion, lectura y codificacion de imagenes, con soporte para multiples formatos incluyendo WebP.

## Estructura del proyecto

* `src/`: Contiene el codigo fuente principal en Rust y los shaders.
  * `main.rs`: Punto de entrada del servidor web y configuracion de rutas.
  * `gpu_compute.rs`: Controladores para inicializar e interactuar con el contexto y los buffers de `wgpu`.
  * `upscaler.rs`: Logica principal del proceso de escalado e interaccion con los shaders.
  * `video.rs`: Logica para procesar e iterar flujos de video.
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
