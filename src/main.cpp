#include "ofMain.h"
#include "surface_network.hpp"

int main(int argc, char ** argv) {
    ofSetupOpenGL(1024, 768, OF_WINDOW);
    ofRunApp(new SurfaceNetworkApp(argc, argv));
}
