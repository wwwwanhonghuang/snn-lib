# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/Micro/Desktop/MainResearch/snn-lib-cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/Micro/Desktop/MainResearch/snn-lib-cpp

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /mnt/c/Users/Micro/Desktop/MainResearch/snn-lib-cpp/CMakeFiles /mnt/c/Users/Micro/Desktop/MainResearch/snn-lib-cpp//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /mnt/c/Users/Micro/Desktop/MainResearch/snn-lib-cpp/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named shared

# Build rule for target.
shared: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 shared
.PHONY : shared

# fast build rule for target.
shared/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/build
.PHONY : shared/fast

#=============================================================================
# Target rules for targets named snn-main

# Build rule for target.
snn-main: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 snn-main
.PHONY : snn-main

# fast build rule for target.
snn-main/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/build
.PHONY : snn-main/fast

#=============================================================================
# Target rules for targets named custom_clean

# Build rule for target.
custom_clean: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 custom_clean
.PHONY : custom_clean

# fast build rule for target.
custom_clean/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/custom_clean.dir/build.make CMakeFiles/custom_clean.dir/build
.PHONY : custom_clean/fast

src/connections/all_to_all_conntection.o: src/connections/all_to_all_conntection.cpp.o
.PHONY : src/connections/all_to_all_conntection.o

# target to build an object file
src/connections/all_to_all_conntection.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/connections/all_to_all_conntection.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/connections/all_to_all_conntection.cpp.o
.PHONY : src/connections/all_to_all_conntection.cpp.o

src/connections/all_to_all_conntection.i: src/connections/all_to_all_conntection.cpp.i
.PHONY : src/connections/all_to_all_conntection.i

# target to preprocess a source file
src/connections/all_to_all_conntection.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/connections/all_to_all_conntection.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/connections/all_to_all_conntection.cpp.i
.PHONY : src/connections/all_to_all_conntection.cpp.i

src/connections/all_to_all_conntection.s: src/connections/all_to_all_conntection.cpp.s
.PHONY : src/connections/all_to_all_conntection.s

# target to generate assembly for a file
src/connections/all_to_all_conntection.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/connections/all_to_all_conntection.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/connections/all_to_all_conntection.cpp.s
.PHONY : src/connections/all_to_all_conntection.cpp.s

src/connections/connection.o: src/connections/connection.cpp.o
.PHONY : src/connections/connection.o

# target to build an object file
src/connections/connection.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/connections/connection.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/connections/connection.cpp.o
.PHONY : src/connections/connection.cpp.o

src/connections/connection.i: src/connections/connection.cpp.i
.PHONY : src/connections/connection.i

# target to preprocess a source file
src/connections/connection.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/connections/connection.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/connections/connection.cpp.i
.PHONY : src/connections/connection.cpp.i

src/connections/connection.s: src/connections/connection.cpp.s
.PHONY : src/connections/connection.s

# target to generate assembly for a file
src/connections/connection.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/connections/connection.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/connections/connection.cpp.s
.PHONY : src/connections/connection.cpp.s

src/interfaces/function.o: src/interfaces/function.cpp.o
.PHONY : src/interfaces/function.o

# target to build an object file
src/interfaces/function.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/interfaces/function.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/interfaces/function.cpp.o
.PHONY : src/interfaces/function.cpp.o

src/interfaces/function.i: src/interfaces/function.cpp.i
.PHONY : src/interfaces/function.i

# target to preprocess a source file
src/interfaces/function.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/interfaces/function.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/interfaces/function.cpp.i
.PHONY : src/interfaces/function.cpp.i

src/interfaces/function.s: src/interfaces/function.cpp.s
.PHONY : src/interfaces/function.s

# target to generate assembly for a file
src/interfaces/function.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/interfaces/function.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/interfaces/function.cpp.s
.PHONY : src/interfaces/function.cpp.s

src/network/initializer/initializer.o: src/network/initializer/initializer.cpp.o
.PHONY : src/network/initializer/initializer.o

# target to build an object file
src/network/initializer/initializer.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/initializer/initializer.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/initializer/initializer.cpp.o
.PHONY : src/network/initializer/initializer.cpp.o

src/network/initializer/initializer.i: src/network/initializer/initializer.cpp.i
.PHONY : src/network/initializer/initializer.i

# target to preprocess a source file
src/network/initializer/initializer.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/initializer/initializer.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/initializer/initializer.cpp.i
.PHONY : src/network/initializer/initializer.cpp.i

src/network/initializer/initializer.s: src/network/initializer/initializer.cpp.s
.PHONY : src/network/initializer/initializer.s

# target to generate assembly for a file
src/network/initializer/initializer.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/initializer/initializer.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/initializer/initializer.cpp.s
.PHONY : src/network/initializer/initializer.cpp.s

src/network/initializer/normal_weight_initializer.o: src/network/initializer/normal_weight_initializer.cpp.o
.PHONY : src/network/initializer/normal_weight_initializer.o

# target to build an object file
src/network/initializer/normal_weight_initializer.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/initializer/normal_weight_initializer.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/initializer/normal_weight_initializer.cpp.o
.PHONY : src/network/initializer/normal_weight_initializer.cpp.o

src/network/initializer/normal_weight_initializer.i: src/network/initializer/normal_weight_initializer.cpp.i
.PHONY : src/network/initializer/normal_weight_initializer.i

# target to preprocess a source file
src/network/initializer/normal_weight_initializer.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/initializer/normal_weight_initializer.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/initializer/normal_weight_initializer.cpp.i
.PHONY : src/network/initializer/normal_weight_initializer.cpp.i

src/network/initializer/normal_weight_initializer.s: src/network/initializer/normal_weight_initializer.cpp.s
.PHONY : src/network/initializer/normal_weight_initializer.s

# target to generate assembly for a file
src/network/initializer/normal_weight_initializer.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/initializer/normal_weight_initializer.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/initializer/normal_weight_initializer.cpp.s
.PHONY : src/network/initializer/normal_weight_initializer.cpp.s

src/network/network.o: src/network/network.cpp.o
.PHONY : src/network/network.o

# target to build an object file
src/network/network.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/network.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/network.cpp.o
.PHONY : src/network/network.cpp.o

src/network/network.i: src/network/network.cpp.i
.PHONY : src/network/network.i

# target to preprocess a source file
src/network/network.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/network.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/network.cpp.i
.PHONY : src/network/network.cpp.i

src/network/network.s: src/network/network.cpp.s
.PHONY : src/network/network.s

# target to generate assembly for a file
src/network/network.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/network.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/network.cpp.s
.PHONY : src/network/network.cpp.s

src/network/network_builder.o: src/network/network_builder.cpp.o
.PHONY : src/network/network_builder.o

# target to build an object file
src/network/network_builder.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/network_builder.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/network_builder.cpp.o
.PHONY : src/network/network_builder.cpp.o

src/network/network_builder.i: src/network/network_builder.cpp.i
.PHONY : src/network/network_builder.i

# target to preprocess a source file
src/network/network_builder.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/network_builder.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/network_builder.cpp.i
.PHONY : src/network/network_builder.cpp.i

src/network/network_builder.s: src/network/network_builder.cpp.s
.PHONY : src/network/network_builder.s

# target to generate assembly for a file
src/network/network_builder.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/network/network_builder.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/network/network_builder.cpp.s
.PHONY : src/network/network_builder.cpp.s

src/neuron_models/lif_neuron.o: src/neuron_models/lif_neuron.cpp.o
.PHONY : src/neuron_models/lif_neuron.o

# target to build an object file
src/neuron_models/lif_neuron.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/neuron_models/lif_neuron.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/neuron_models/lif_neuron.cpp.o
.PHONY : src/neuron_models/lif_neuron.cpp.o

src/neuron_models/lif_neuron.i: src/neuron_models/lif_neuron.cpp.i
.PHONY : src/neuron_models/lif_neuron.i

# target to preprocess a source file
src/neuron_models/lif_neuron.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/neuron_models/lif_neuron.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/neuron_models/lif_neuron.cpp.i
.PHONY : src/neuron_models/lif_neuron.cpp.i

src/neuron_models/lif_neuron.s: src/neuron_models/lif_neuron.cpp.s
.PHONY : src/neuron_models/lif_neuron.s

# target to generate assembly for a file
src/neuron_models/lif_neuron.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/neuron_models/lif_neuron.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/neuron_models/lif_neuron.cpp.s
.PHONY : src/neuron_models/lif_neuron.cpp.s

src/neuron_models/neuron.o: src/neuron_models/neuron.cpp.o
.PHONY : src/neuron_models/neuron.o

# target to build an object file
src/neuron_models/neuron.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/neuron_models/neuron.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/neuron_models/neuron.cpp.o
.PHONY : src/neuron_models/neuron.cpp.o

src/neuron_models/neuron.i: src/neuron_models/neuron.cpp.i
.PHONY : src/neuron_models/neuron.i

# target to preprocess a source file
src/neuron_models/neuron.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/neuron_models/neuron.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/neuron_models/neuron.cpp.i
.PHONY : src/neuron_models/neuron.cpp.i

src/neuron_models/neuron.s: src/neuron_models/neuron.cpp.s
.PHONY : src/neuron_models/neuron.s

# target to generate assembly for a file
src/neuron_models/neuron.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/neuron_models/neuron.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/neuron_models/neuron.cpp.s
.PHONY : src/neuron_models/neuron.cpp.s

src/recorder/weight_recorder.o: src/recorder/weight_recorder.cpp.o
.PHONY : src/recorder/weight_recorder.o

# target to build an object file
src/recorder/weight_recorder.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/recorder/weight_recorder.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/recorder/weight_recorder.cpp.o
.PHONY : src/recorder/weight_recorder.cpp.o

src/recorder/weight_recorder.i: src/recorder/weight_recorder.cpp.i
.PHONY : src/recorder/weight_recorder.i

# target to preprocess a source file
src/recorder/weight_recorder.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/recorder/weight_recorder.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/recorder/weight_recorder.cpp.i
.PHONY : src/recorder/weight_recorder.cpp.i

src/recorder/weight_recorder.s: src/recorder/weight_recorder.cpp.s
.PHONY : src/recorder/weight_recorder.s

# target to generate assembly for a file
src/recorder/weight_recorder.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/recorder/weight_recorder.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/recorder/weight_recorder.cpp.s
.PHONY : src/recorder/weight_recorder.cpp.s

src/simulator/snn_simulator.o: src/simulator/snn_simulator.cpp.o
.PHONY : src/simulator/snn_simulator.o

# target to build an object file
src/simulator/snn_simulator.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/simulator/snn_simulator.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/simulator/snn_simulator.cpp.o
.PHONY : src/simulator/snn_simulator.cpp.o

src/simulator/snn_simulator.i: src/simulator/snn_simulator.cpp.i
.PHONY : src/simulator/snn_simulator.i

# target to preprocess a source file
src/simulator/snn_simulator.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/simulator/snn_simulator.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/simulator/snn_simulator.cpp.i
.PHONY : src/simulator/snn_simulator.cpp.i

src/simulator/snn_simulator.s: src/simulator/snn_simulator.cpp.s
.PHONY : src/simulator/snn_simulator.s

# target to generate assembly for a file
src/simulator/snn_simulator.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/simulator/snn_simulator.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/simulator/snn_simulator.cpp.s
.PHONY : src/simulator/snn_simulator.cpp.s

src/snn-main.o: src/snn-main.cpp.o
.PHONY : src/snn-main.o

# target to build an object file
src/snn-main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/snn-main.cpp.o
.PHONY : src/snn-main.cpp.o

src/snn-main.i: src/snn-main.cpp.i
.PHONY : src/snn-main.i

# target to preprocess a source file
src/snn-main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/snn-main.cpp.i
.PHONY : src/snn-main.cpp.i

src/snn-main.s: src/snn-main.cpp.s
.PHONY : src/snn-main.s

# target to generate assembly for a file
src/snn-main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/snn-main.cpp.s
.PHONY : src/snn-main.cpp.s

src/synapse_models/synapse.o: src/synapse_models/synapse.cpp.o
.PHONY : src/synapse_models/synapse.o

# target to build an object file
src/synapse_models/synapse.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/synapse_models/synapse.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/synapse_models/synapse.cpp.o
.PHONY : src/synapse_models/synapse.cpp.o

src/synapse_models/synapse.i: src/synapse_models/synapse.cpp.i
.PHONY : src/synapse_models/synapse.i

# target to preprocess a source file
src/synapse_models/synapse.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/synapse_models/synapse.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/synapse_models/synapse.cpp.i
.PHONY : src/synapse_models/synapse.cpp.i

src/synapse_models/synapse.s: src/synapse_models/synapse.cpp.s
.PHONY : src/synapse_models/synapse.s

# target to generate assembly for a file
src/synapse_models/synapse.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/shared.dir/build.make CMakeFiles/shared.dir/src/synapse_models/synapse.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/snn-main.dir/build.make CMakeFiles/snn-main.dir/src/synapse_models/synapse.cpp.s
.PHONY : src/synapse_models/synapse.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... custom_clean"
	@echo "... shared"
	@echo "... snn-main"
	@echo "... src/connections/all_to_all_conntection.o"
	@echo "... src/connections/all_to_all_conntection.i"
	@echo "... src/connections/all_to_all_conntection.s"
	@echo "... src/connections/connection.o"
	@echo "... src/connections/connection.i"
	@echo "... src/connections/connection.s"
	@echo "... src/interfaces/function.o"
	@echo "... src/interfaces/function.i"
	@echo "... src/interfaces/function.s"
	@echo "... src/network/initializer/initializer.o"
	@echo "... src/network/initializer/initializer.i"
	@echo "... src/network/initializer/initializer.s"
	@echo "... src/network/initializer/normal_weight_initializer.o"
	@echo "... src/network/initializer/normal_weight_initializer.i"
	@echo "... src/network/initializer/normal_weight_initializer.s"
	@echo "... src/network/network.o"
	@echo "... src/network/network.i"
	@echo "... src/network/network.s"
	@echo "... src/network/network_builder.o"
	@echo "... src/network/network_builder.i"
	@echo "... src/network/network_builder.s"
	@echo "... src/neuron_models/lif_neuron.o"
	@echo "... src/neuron_models/lif_neuron.i"
	@echo "... src/neuron_models/lif_neuron.s"
	@echo "... src/neuron_models/neuron.o"
	@echo "... src/neuron_models/neuron.i"
	@echo "... src/neuron_models/neuron.s"
	@echo "... src/recorder/weight_recorder.o"
	@echo "... src/recorder/weight_recorder.i"
	@echo "... src/recorder/weight_recorder.s"
	@echo "... src/simulator/snn_simulator.o"
	@echo "... src/simulator/snn_simulator.i"
	@echo "... src/simulator/snn_simulator.s"
	@echo "... src/snn-main.o"
	@echo "... src/snn-main.i"
	@echo "... src/snn-main.s"
	@echo "... src/synapse_models/synapse.o"
	@echo "... src/synapse_models/synapse.i"
	@echo "... src/synapse_models/synapse.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

