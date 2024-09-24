import 'dart:io' as io;
import 'package:flutter/material.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:ultralytics_yolo/yolo_model.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  final controller = UltralyticsYoloCameraController();

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('YOLO Object Detection')),
        body: FutureBuilder<bool>(
          future: _checkPermissions(),
          builder: (context, snapshot) {
            final allPermissionsGranted = snapshot.data ?? false;

            if (!allPermissionsGranted) {
              return Center(child: const Text('Please grant camera and storage permissions.'));
            }

            return FutureBuilder<ObjectDetector>(
              future: _initObjectDetectorWithLocalModel(),
              builder: (context, snapshot) {
                final predictor = snapshot.data;

                if (predictor == null) {
                  return const Center(child: CircularProgressIndicator());
                }

                return Stack(
                  children: [
                    UltralyticsYoloCameraPreview(
                      controller: controller,
                      predictor: predictor,
                      onCameraCreated: () {
                        predictor.loadModel(useGpu: false);
                      },
                    ),
                    StreamBuilder<double?>(
                      stream: predictor.inferenceTime,
                      builder: (context, snapshot) {
                        final inferenceTime = snapshot.data;

                        return StreamBuilder<double?>(
                          stream: predictor.fpsRate,
                          builder: (context, snapshot) {
                            final fpsRate = snapshot.data;

                            return Times(
                              inferenceTime: inferenceTime,
                              fpsRate: fpsRate,
                            );
                          },
                        );
                      },
                    ),
                  ],
                );
              },
            );
          },
        ),
        floatingActionButton: FloatingActionButton(
          child: const Icon(Icons.camera),
          onPressed: () {
            controller.toggleLensDirection();
          },
        ),
      ),
    );
  }

  Future<ObjectDetector> _initObjectDetectorWithLocalModel() async {
    // Detect the platform and load the correct model file format
    final String modelPath;
    final String metadataPath;

    if (io.Platform.isAndroid) {
      modelPath = await _copy('assets/yolo_v8_tomato_int8.tflite');
      metadataPath = await _copy('assets/metadata_tomato.yaml');
    } else if (io.Platform.isIOS) {
      modelPath = await _copy('assets/yolo_v8_tomato.mlmodel');
      metadataPath = await _copy('assets/metadata_tomato.yaml');
    } else {
      throw UnsupportedError('This platform is not supported.');
    }

    // Initialize the YOLO model for object detection
    final model = LocalYoloModel(
      id: '',
      task: Task.detect,
      format: io.Platform.isIOS ? Format.coreml : Format.tflite,
      modelPath: modelPath,
      metadataPath: metadataPath,
    );

    return ObjectDetector(model: model);
  }

  Future<String> _copy(String assetPath) async {
    final path = '${(await getApplicationSupportDirectory()).path}/$assetPath';
    await io.Directory(dirname(path)).create(recursive: true);
    final file = io.File(path);

    if (!await file.exists()) {
      final byteData = await rootBundle.load(assetPath);
      await file.writeAsBytes(byteData.buffer
          .asUint8List(byteData.offsetInBytes, byteData.lengthInBytes));
    }
    return file.path;
  }

  Future<bool> _checkPermissions() async {
    List<Permission> permissions = [];

    if (await Permission.camera.request().isDenied) {
      permissions.add(Permission.camera);
    }
    if (await Permission.storage.request().isDenied) {
      permissions.add(Permission.storage);
    }

    return permissions.isEmpty;
  }
}

class Times extends StatelessWidget {
  const Times({
    super.key,
    required this.inferenceTime,
    required this.fpsRate,
  });

  final double? inferenceTime;
  final double? fpsRate;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Align(
        alignment: Alignment.bottomCenter,
        child: Container(
          margin: const EdgeInsets.all(20),
          padding: const EdgeInsets.all(20),
          decoration: const BoxDecoration(
            borderRadius: BorderRadius.all(Radius.circular(10)),
            color: Colors.black54,
          ),
          child: Text(
            '${(inferenceTime ?? 0).toStringAsFixed(1)} ms  -  ${(fpsRate ?? 0).toStringAsFixed(1)} FPS',
            style: const TextStyle(color: Colors.white70),
          ),
        ),
      ),
    );
  }
}
