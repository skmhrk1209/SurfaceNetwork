#include <tensorflow/core/platform/init_main.h>
#include <tensorflow/core/public/session.h>
#include "ofMain.h"
#include "iostream.hpp"

namespace tf = tensorflow;

std::unique_ptr<tf::Session> createGraph(const std::string &filename)
{
    tf::GraphDef graphDef;
    auto status = tf::ReadBinaryProto(tf::Env::Default(), filename, &graphDef);
    if (status.ok())
    {
        std::unique_ptr<tf::Session> session(tf::NewSession(tf::SessionOptions()));
        auto status = session->Create(graphDef);
        if (!status.ok())
        {
            return session;
        }
        else
        {
            LOG(ERROR) << status.error_message();
        }
    }
    else
    {
        LOG(ERROR) << status.error_message();
    }
}

std::vector<tf::Tensor> forward(const std::unique_ptr<tf::Session> &session,
                                const std::vector<std::pair<std::string, tf::Tensor>> &inputs,
                                const std::vector<std::string> &output_names)
{
    std::vector<tf::Tensor> outputs;
    auto status = session->Run(inputs, output_names, {}, &outputs);
    if (!status.ok())
    {
        LOG(ERROR) << status.error_message();
    }
    return outputs;
}


class SurfaceNetworkApp : public ofBaseApp
{
public:
	SurfaceNetworkApp(int, char **);

	void setup() override;
	void update() override;
	void draw() override;

	void generateSurface();

	std::unique_ptr<tf::Session> mSession;

	ofPlanePrimitive mGrid;
	of3dPrimitive mSurface;

	vector<ofVec2f> mVertices;

	ofEasyCam mCamera;
	ofLight mLight;
	ofShader mShader;
};

SurfaceNetworkApp::SurfaceNetworkApp(int argc, char **argv)
{
	tf::port::InitMain(argv[0], &argc, &argv);
}

void SurfaceNetworkApp::setup()
{
	mShader.load("", "/Users/sakuma/Documents/openFrameworks-0.10.1/apps/myApps/SurfaceNetwork/src/shader.flag");

	mGrid.set(1000, 1000);
	mGrid.setResolution(100, 100);
	mSurface.getMesh().setMode(mGrid.getMesh().getMode());
	mSurface.getMesh().addIndices(mGrid.getMesh().getIndices());

	mSession = createGraph("/Users/sakuma/Documents/openFrameworks-0.10.1/apps/myApps/SurfaceNetwork/src/graph.pb");

	ofBackground(0);
	ofEnableDepthTest();
	ofEnableSmoothing();
}

void SurfaceNetworkApp::update()
{
	generateSurface();

	mVertices.clear();
	for (const auto &world : mSurface.getMesh().getVertices())
	{
		auto screen = mCamera.worldToScreen(world);
		mVertices.emplace_back(screen.x, ofMap(screen.y, 0, ofGetHeight(), ofGetHeight(), 0));
	}
}

void SurfaceNetworkApp::draw()
{
	mCamera.begin();
	// mLight.enable();
	ofSetColor(255);
	ofSetLineWidth(0.1);
	mSurface.drawWireframe();
	// mLight.disable();
	mCamera.end();

	mShader.begin();
	mShader.setUniform2fv("vertices", &mVertices[0][0], mVertices.size());
	ofSetColor(255);
	ofDrawRectangle(0, 0, ofGetWidth(), ofGetHeight());
	mShader.end();

	ofDrawBitmapString(ofToString(ofGetFrameRate()) + "fps", 20, 20);
}

void SurfaceNetworkApp::generateSurface()
{
	mSurface.getMesh().clear();
    
    const auto& gridVertices = mGrid.getMesh().getVertices();
    auto gridTensor = tf::Tensor(tf::DT_FLOAT, tf::TensorShape({static_cast<long long>(gridVertices.size()), 3}));
    auto gridData = gridTensor.tensor<float, 2>();
    auto z = ofSignedNoise(ofGetElapsedTimef());
    for (auto i = 0; i < gridVertices.size(); ++i)
    {
        gridData(i, 0) = gridVertices[i].x;
        gridData(i, 1) = gridVertices[i].y;
        gridData(i, 2) = z;
    }
	
    std::vector<glm::vec3> surfaceVertices;
    std::vector<ofFloatColor> surfaceColors;
	auto surfaceTensor = forward(mSession, {{"x", gridTensor}}, {"y"})[0];
    auto surfaceData = surfaceTensor.tensor<float, 2>();
    for (auto i = 0; i < surfaceData.dimension(0); ++i)
    {
        surfaceVertices.emplace_back(glm::vec3(surfaceData(i, 0), surfaceData(i, 1), surfaceData(i, 2)) * 100);
        surfaceColors.emplace_back(ofNoise(surfaceVertices.back()));
    }
	mSurface.getMesh().addVertices(surfaceVertices);
	mSurface.getMesh().addColors(surfaceColors);

	const auto &surfaceIndices = mSurface.getMesh().getIndices();

	std::vector<glm::vec3> surfaceNormals;
	for (auto i = 0; i + 2 < surfaceIndices.size(); ++i)
	{
		surfaceNormals.emplace_back(glm::cross(surfaceVertices[surfaceIndices[i + 1]] - surfaceVertices[surfaceIndices[i]], surfaceVertices[surfaceIndices[i + 2]] - surfaceVertices[surfaceIndices[i]]));
	}
	mSurface.getMesh().addNormals(surfaceNormals);
}
