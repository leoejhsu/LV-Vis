# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

from vispy.visuals.volume import VolumeVisual
from vispy.visuals.shaders import Function
from vispy.visuals import Visual

_VERTEX_SHADER = """
attribute vec3 a_position;
uniform vec3 u_shape;

varying vec3 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

void main() {
    v_position = a_position;

    // Project local vertex coordinate to camera position. Then do a step
    // backward (in cam coords) and project back. Voila, we get our ray vector.
    vec4 pos_in_cam = $viewtransformf(vec4(v_position, 1));

    // intersection of ray and near clipping plane (z = -1 in clip coords)
    pos_in_cam.z = -pos_in_cam.w;
    v_nearpos = $viewtransformi(pos_in_cam);

    // intersection of ray and far clipping plane (z = +1 in clip coords)
    pos_in_cam.z = pos_in_cam.w;
    v_farpos = $viewtransformi(pos_in_cam);

    gl_Position = $transform(vec4(v_position, 1.0));
}
"""  # noqa

_FRAGMENT_SHADER1 = """
// uniforms
uniform $sampler_type u_volumetex;
uniform vec3 u_shape;
uniform vec2 clim;
uniform float gamma;
uniform float u_threshold;
uniform float u_attenuation;
uniform float u_relative_step_size;
uniform float u_mip_cutoff;
uniform float u_minip_cutoff;

uniform float u_min_val; 
uniform float u_max_val;  


//varyings
varying vec3 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

// uniforms for lighting. Hard coded until we figure out how to do lights
const vec4 u_ambient = vec4(0.2, 0.2, 0.2, 1.0);
const vec4 u_diffuse = vec4(0.8, 0.2, 0.2, 1.0);
const vec4 u_specular = vec4(1.0, 1.0, 1.0, 1.0);
const float u_shininess = 40.0;

// uniforms for plane definition. Defined in data coordinates.
uniform vec3 u_plane_normal;
uniform vec3 u_plane_position;
uniform float u_plane_thickness;

//varying vec3 lightDirs[1];

// global holding view direction in local coordinates
vec3 view_ray;



vec4 applyColormap(float data) {
    data = clamp(data, min(clim.x, clim.y), max(clim.x, clim.y));
    data = (data - clim.x) / (clim.y - clim.x);
    vec4 color = $cmap(pow(data, gamma));
    return color;
}


void main() {
    vec3 farpos = v_farpos.xyz / v_farpos.w;
    vec3 nearpos = v_nearpos.xyz / v_nearpos.w;

    // Calculate unit vector pointing in the view direction through this
    // fragment.
    view_ray = normalize(farpos.xyz - nearpos.xyz);

    // Variables to keep track of where to set the frag depth.
    // frag_depth_point is in data coordinates.
    vec3 frag_depth_point;

    // Set up the ray casting
    // This snippet must define three variables:
    // vec3 start_loc - the starting location of the ray in texture coordinates
    // vec3 step - the step vector in texture coordinates
    // int nsteps - the number of steps to make through the texture

    $raycasting_setup

    // For testing: show the number of steps. This helps to establish
    // whether the rays are correctly oriented
    //gl_FragColor = vec4(0.0, f_nsteps / 3.0 / u_shape.x, 1.0, 1.0);
    //return;

    $before_loop

    // This outer loop seems necessary on some systems for large
    // datasets. Ugly, but it works ...
    vec3 loc = start_loc;
    int iter = 0;

    // keep track if the texture is ever sampled; if not, fragment will be discarded
    // this allows us to discard fragments that only traverse clipped parts of the texture
    bool texture_sampled = false;

    while (iter < nsteps) {
        for (iter=iter; iter<nsteps; iter++)
        {
            // Only sample volume if loc is not clipped by clipping planes
            float distance_from_clip = $clip_with_planes(loc, u_shape);

            if (distance_from_clip >= 0)
            {
                // Get sample color
                vec4 color = $get_data(loc);
                float val = color.r;
                texture_sampled = true;

                $in_loop
            }
            // Advance location deeper into the volume
            loc += step;
        }
    }

    if (!texture_sampled)
        discard;

    $after_loop

    // set frag depth
    vec4 frag_depth_vector = vec4(frag_depth_point, 1);
    vec4 iproj = $viewtransformf(frag_depth_vector);
    iproj.z /= iproj.w;
    gl_FragDepth = (iproj.z+1.0)/2.0;
}
"""  # noqa

_shaders = {
    'vertex': _VERTEX_SHADER,
    'fragment': _FRAGMENT_SHADER1,
}

_MIP_SNIPPETS = dict(
    before_loop="""
        float maxval = u_mip_cutoff; // The maximum encountered value
        int maxi = -1;  // Where the maximum value was encountered
        """,
    in_loop="""
        if (val < u_min_val || val > u_max_val) {
            val = 0.0;
        } 
        if ( val > maxval ) {
            maxval = val;
            maxi = iter;
            if ( maxval >= clim.y ) {
                // stop if no chance of finding a higher maxval
                iter = nsteps;
            }
        }
        """,
    after_loop="""
        // Refine search for max value, but only if anything was found
        if ( maxi > -1 ) {
            // Calculate starting location of ray for sampling
            vec3 start_loc_refine = start_loc + step * (float(maxi) - 0.5);
            loc = start_loc_refine;

            // Variables to keep track of current value and where max was encountered
            vec3 max_loc_tex = start_loc_refine;

            vec3 small_step = step * 0.1;
            for (int i=0; i<10; i++) {
                float val = $get_data(loc).r;     
                if (val < u_min_val || val > u_max_val) {
                    val = 0.0;
                    // loc += small_step; 
                    // continue;
                }        
                // if (val >= u_min_val && val <= u_max_val && val > maxval) {
                if ( val > maxval) {
                    maxval = val;
                    max_loc_tex = start_loc_refine + (small_step * i);
                }
                loc += small_step;
            }
            frag_depth_point = max_loc_tex * u_shape;

            // gl_FragColor = vec4(max_loc_tex.x, max_loc_tex.y, max_loc_tex.z, 1);
            // if (u_min_val > maxval || maxval > u_max_val)
                // maxval = 0.0;
            gl_FragColor = applyColormap(maxval);


            /* if (maxval < u_min_val ) {
                gl_FragColor = applyColormap(u_min_val);
            } else if (maxval > u_max_val){
                gl_FragColor = applyColormap(u_max_val);
            } else {
               gl_FragColor = applyColormap(maxval);
            } */

        } else {
            discard;
        }
        """,

)


# gl_FragColor = vec4(applyColormap(maxval)[0], max_loc_tex.x, max_loc_tex.y, max_loc_tex.z);
class MyVisual(VolumeVisual):
    def __init__(self, vol, clim="auto", method='mip', threshold=None,
                 attenuation=1.0, relative_step_size=0.8, cmap='grays',
                 gamma=1.0, interpolation='linear', texture_format=None,
                 raycasting_mode='volume', plane_position=None,
                 plane_normal=None, plane_thickness=1.0, clipping_planes=None,
                 clipping_planes_coord_system='scene', mip_cutoff=None,
                 minip_cutoff=None):
        super().__init__(vol=vol, clim=clim, method=method, threshold=threshold,
                         attenuation=attenuation, relative_step_size=relative_step_size,
                         cmap=cmap, gamma=gamma, interpolation=interpolation,
                         texture_format=texture_format, raycasting_mode=raycasting_mode,
                         plane_position=plane_position, plane_normal=plane_normal,
                         plane_thickness=plane_thickness, clipping_planes=clipping_planes,
                         clipping_planes_coord_system=clipping_planes_coord_system,
                         mip_cutoff=mip_cutoff, minip_cutoff=minip_cutoff)

        self._rendering_methods = {
            'mip': _MIP_SNIPPETS
            # Add other methods as needed
        }
        self.method = method
        self.shared_program.vert = _shaders['vertex']  # Vertex shader
        self.shared_program.frag = _shaders['fragment']  # Fragment shader
        self._texture.interpolation = 'linear'
        # print('clim: ',self.shared_program['clim'])
        # print('gamma:', self.shared_program['gamma'])

        # self.set_data(vol, clim or "auto", False)

    @property
    def method(self):
        """The render method to use

        Current options are:

            * translucent: voxel colors are blended along the view ray until
              the result is opaque.
            * mip: maxiumum intensity projection. Cast a ray and display the
              maximum value that was encountered.
            * minip: minimum intensity projection. Cast a ray and display the
              minimum value that was encountered.
            * attenuated_mip: attenuated maximum intensity projection. Cast a
              ray and display the maximum value encountered. Values are
              attenuated as the ray moves deeper into the volume.
            * additive: voxel colors are added along the view ray until
              the result is saturated.
            * iso: isosurface. Cast a ray until a certain threshold is
              encountered. At that location, lighning calculations are
              performed to give the visual appearance of a surface.
            * average: average intensity projection. Cast a ray and display the
              average of values that were encountered.
        """
        # print('frag: ', self.shared_program.frag)
        return self._method

    @method.setter
    def method(self, method):
        # Check and save
        known_methods = list(self._rendering_methods.keys())
        if method not in known_methods:
            raise ValueError('Volume render method should be in %r, not %r' %
                             (known_methods, method))
        self._method = method

        # $get_data needs to be unset and re-set, since it's present inside the snippets.
        #       Program should probably be able to do this automatically
        self.shared_program.frag['get_data'] = None
        self.shared_program.frag['raycasting_setup'] = self._raycasting_setup_snippet
        self.shared_program.frag['before_loop'] = self._before_loop_snippet
        self.shared_program.frag['in_loop'] = self._in_loop_snippet
        self.shared_program.frag['after_loop'] = self._after_loop_snippet
        self.shared_program.frag['sampler_type'] = self._texture.glsl_sampler_type
        self.shared_program.frag['cmap'] = Function(self._cmap.glsl_map)
        self.shared_program['texture2D_LUT'] = self.cmap.texture_lut()
        self.shared_program['u_mip_cutoff'] = self._mip_cutoff
        self.shared_program['u_minip_cutoff'] = self._minip_cutoff

        self._need_interpolation_update = True
        self.update()

    @property
    def _raycasting_setup_snippet(self):
        return self._raycasting_modes[self.raycasting_mode]

    @property
    def _before_loop_snippet(self):
        return self._rendering_methods[self.method]['before_loop']

    @property
    def _in_loop_snippet(self):
        return self._rendering_methods[self.method]['in_loop']

    @property
    def _after_loop_snippet(self):
        return self._rendering_methods[self.method]['after_loop']
