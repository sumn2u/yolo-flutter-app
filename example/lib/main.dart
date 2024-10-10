import 'dart:io' as io;
import 'package:flutter/material.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';
import 'package:image_picker/image_picker.dart';
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
  final ImagePicker _picker = ImagePicker();
  io.File? _selectedImage;
  List<ClassificationResult>? _classificationResults;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(title: const Text('YOLO Object Classification')),
        body: FutureBuilder<bool>(
          future: _checkPermissions(),
          builder: (context, snapshot) {
            final allPermissionsGranted = snapshot.data ?? false;

            if (!allPermissionsGranted) {
              return const Center(child: Text('Please grant camera and storage permissions.'));
            }

            return FutureBuilder<ImageClassifier>(
              future: _initObjectClassifierWithLocalModel(),
              builder: (context, snapshot) {
                final classifier = snapshot.data;

                if (classifier == null) {
                  return const Center(child: CircularProgressIndicator());
                }

                return Column(
                  children: [
                    Expanded(
                      child: Stack(
                        children: [
                          _selectedImage == null
                              ? const Center(child: Text('No image selected'))
                              : Image.file(_selectedImage!), // Display selected image
                          if (_classificationResults != null)
                            Positioned(
                              bottom: 20,
                              left: 20,
                              child: ClassificationResultsWidget(results: _classificationResults!),
                            ),
                        ],
                      ),
                    ),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        FloatingActionButton(
                          heroTag: 'pick_image',
                          child: const Icon(Icons.add_a_photo),
                          onPressed: () {
                            _showImageSourceDialog(context);
                          },
                        ),
                      ],
                    ),
                  ],
                );
              },
            );
          },
        ),
      ),
    );
  }

  Future<ImageClassifier> _initObjectClassifierWithLocalModel() async {
    // Detect the platform and load the correct model file format
    final String modelPath;
    final String metadataPath;

    if (io.Platform.isAndroid) {
      modelPath = await _copy('assets/yolo_v8_tomato_int8.tflite');
      metadataPath = await _copy('assets/metadata_tomato_cls.yaml');
    } else if (io.Platform.isIOS) {
      modelPath = await _copy('assets/yolo_v8_tomato.mlmodel');
      metadataPath = await _copy('assets/metadata_tomato_cls.yaml');
    } else {
      throw UnsupportedError('This platform is not supported.');
    }

    // Initialize the YOLO model for object classification
    final model = LocalYoloModel(
      id: '',
      task: Task.classify, // Set task to classification
      format: io.Platform.isIOS ? Format.coreml : Format.tflite,
      modelPath: modelPath,
      metadataPath: metadataPath,
    );

    return ImageClassifier(model: model);
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

  Future<void> _showImageSourceDialog(context) async {
    final classifier = await _initObjectClassifierWithLocalModel();
  
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: const Text("Select Image Source"),
          actions: [
            TextButton(
              child: const Text("Camera"),
              onPressed: () async {
                Navigator.of(context).pop(); // Close the dialog
                final pickedFile = await _picker.pickImage(source: ImageSource.camera);
                if (pickedFile != null) {
                  setState(() {
                    _selectedImage = io.File(pickedFile.path);
                  });
                  // Classify the captured image
                  final results = await classifier.classify(imagePath: _selectedImage!.path);
                  setState(() {
                    _classificationResults = results?.whereType<ClassificationResult>().toList();
                  });
                }
              },
            ),
            TextButton(
              child: const Text("Gallery"),
              onPressed: () async {
                Navigator.of(context).pop(); // Close the dialog
                await _pickImageFromGallery(classifier);
              },
            ),
          ],
        );
      },
    );
  }

  Future<void> _pickImageFromGallery(ImageClassifier classifier) async {
    // Pick an image from the gallery
    final pickedFile = await _picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _selectedImage = io.File(pickedFile.path);
      });
      // Classify the selected image
      final results = await classifier.classify(imagePath: _selectedImage!.path);
      setState(() {
        _classificationResults = results?.whereType<ClassificationResult>().toList();
      });
    }
  }
}

class ClassificationResultsWidget extends StatelessWidget {
  final List<ClassificationResult> results;

  const ClassificationResultsWidget({Key? key, required this.results}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: results.map((result) {
          return Text(
            '${result.label}: ${(result.confidence * 100).toStringAsFixed(2)}%',
            style: const TextStyle(color: Colors.white),
          );
        }).toList(),
      ),
    );
  }
}
