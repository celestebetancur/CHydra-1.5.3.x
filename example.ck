/* --------------------------------------------------------------------------------

-------------------------------------------------------------------------------- */

// Basic shader compilation, it runs once using the CHydra wrapper to create WGSL code

@import "CHydra"

// turn off tonemapping
GG.outputPass().tonemap(OutputPass.ToneMap_None);
GG.windowTitle( "CHydra" );
11 => GG.camera().posZ;

Webcam webcam;

Material shader_material;
GPlane plane --> GG.scene();

plane.mat(shader_material);
plane.scaX(16);
plane.scaY(9.1);
plane.rot(@(0,0,Math.PI));

// set default sampler and texture
TextureSampler default_sampler;

shader_material.sampler(1, default_sampler);
shader_material.texture(2, webcam.texture());
shader_material.texture(3, webcam.texture());

CHydra st;
global string ShaderCode;
0 => global int recompile;
ShaderDesc shader_desc;

fun void recompileShader() {
    st.shader(ShaderCode) => string processed_shader_code;

    processed_shader_code => shader_desc.vertexCode;
    processed_shader_code => shader_desc.fragmentCode;

    Shader custom_shader(shader_desc); // create shader from shader_desc
    custom_shader => shader_material.shader; // connect shader to material
}

(
    st.osc(13,1,1)
).code => ShaderCode;

recompileShader();

while (true) {
    // set time
    plane.mat().uniformFloat(0, now/second);
    // The time is now
    GG.nextFrame() => now;

    // if you comment this then the shader runs
    if(recompile){
        recompileShader();
        0 => recompile;
    }
}

