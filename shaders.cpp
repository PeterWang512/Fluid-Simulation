/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#define STRINGIFY(A) #A

// vertex shader
const char *vertexShader = STRINGIFY(
                               uniform float pointRadius;  // point size in world space
                               uniform float pointScale;   // scale to calculate size in pixels
                               uniform float densityScale;
                               uniform float densityOffset;
                               void main()
{
    // calculate window-space point size
    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);

    gl_TexCoord[0] = gl_MultiTexCoord0;
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);

    gl_FrontColor = gl_Color;
}
                           );

// pixel shader for rendering points as shaded spheres
const char *spherePixelShader = STRINGIFY(
                                    void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0-mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

    gl_FragColor = gl_Color * diffuse;

	/*float near = 0.01;
	float far = 4.0;
	float z = gl_FragCoord.z * 2.0 - 1.0;
	float depth = (2.0 * near * far) / (near + far - z * (far - near));
	gl_FragColor = vec4(vec3(0.0, 0.0, depth), 1.0);*/
}
                                );

const char *depthShader = STRINGIFY(
	uniform sampler2D depthTex;
	void main()
{
	const vec3 lightDir = vec3(0.577, 0.577, 0.577);
	// calculate normal from texture coordinates
	vec3 N;
	N.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
	float mag = dot(N.xy, N.xy);

	if (mag > 1.0) discard;   // kill pixels outside circle

	N.z = sqrt(1.0 - mag);

	vec4 eyeSpacePos = vec4(gl_TexCoord[1].xyz + N * pointRadius, 1.0);
	vec4 clipSpacePos = gl_ProjectionMatrix * eyeSpacePos;
	gl_FragDepth = (clipSpacePos.z / clipSpacePos.w)*0.5 + 0.5;
	float diffuse = max(0.0, dot(N, lightDir));
	gl_FragColor = diffuse * gl_Color;
}
										);

const char *passThruVS = STRINGIFY(
	void main()
	{
		gl_Position = gl_Vertex;                                       
		gl_TexCoord[0] = gl_MultiTexCoord0;                            
		gl_FrontColor = gl_Color;                                      
	}                                                                  
);

const char *texture2DPS = STRINGIFY(
	uniform sampler2D tex;                                             
	void main()                                                        
	{
		gl_FragColor = texture2D(tex, gl_TexCoord[0].xy);              
	}                                                                  
);

// 4 tap 3x3 gaussian blur
const char *blurPS = STRINGIFY(
	uniform sampler2D tex;                                                                
	uniform vec2 texelSize;                                                               
	uniform float blurRadius;                                                             
	void main()                                                                           
	{

		vec4 c;                                                                           
		c = texture2D(tex, gl_TexCoord[0].xy + vec2(-0.5, -0.5)*texelSize*blurRadius);    
		c += texture2D(tex, gl_TexCoord[0].xy + vec2(0.5, -0.5)*texelSize*blurRadius);    
		c += texture2D(tex, gl_TexCoord[0].xy + vec2(0.5, 0.5)*texelSize*blurRadius);     
		c += texture2D(tex, gl_TexCoord[0].xy + vec2(-0.5, 0.5)*texelSize*blurRadius);    
		c *= 0.25;                                                                        

		gl_FragColor = c;                                                                 
	}                                                                                     
);
