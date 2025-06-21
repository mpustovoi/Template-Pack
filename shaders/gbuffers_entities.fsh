#version 450 compatibility

uniform sampler2D lightmap;
uniform sampler2D gtexture;
uniform vec4 entityColor;

in vec2 lmcoord;
in vec2 texcoord;
in vec4 glcolor;
in vec3 Normal;

/* DRAWBUFFERS:057 */
layout(location = 0) out vec4 color;
layout(location = 1) out vec4 ScreenNormals;
layout(location = 2) out vec4 Blocklight;

void main() {
	color = texture(gtexture, texcoord) * glcolor;
	color.rgb = mix(color.rgb, entityColor.rgb, entityColor.a);
	Blocklight = texture(lightmap, lmcoord)-.6;
    ScreenNormals = vec4((Normal + 1.) / 2., 1);
    
	if (color.a < 0.1) {
		discard;
	}
}