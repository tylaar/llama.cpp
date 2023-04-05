# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/yifengyu/hack/llama.cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/yifengyu/hack/llama.cpp

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/local/bin/ccmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/local/bin/cmake --regenerate-during-build -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/yifengyu/hack/llama.cpp/CMakeFiles /Users/yifengyu/hack/llama.cpp//CMakeFiles/progress.marks
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/yifengyu/hack/llama.cpp/CMakeFiles 0
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
# Target rules for targets named llama

# Build rule for target.
llama: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 llama
.PHONY : llama

# fast build rule for target.
llama/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/build
.PHONY : llama/fast

#=============================================================================
# Target rules for targets named quantize

# Build rule for target.
quantize: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 quantize
.PHONY : quantize

# fast build rule for target.
quantize/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/build
.PHONY : quantize/fast

#=============================================================================
# Target rules for targets named ggml

# Build rule for target.
ggml: cmake_check_build_system
	$(MAKE) $(MAKESILENT) -f CMakeFiles/Makefile2 ggml
.PHONY : ggml

# fast build rule for target.
ggml/fast:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/build
.PHONY : ggml/fast

ggml.o: ggml.c.o
.PHONY : ggml.o

# target to build an object file
ggml.c.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/ggml.c.o
.PHONY : ggml.c.o

ggml.i: ggml.c.i
.PHONY : ggml.i

# target to preprocess a source file
ggml.c.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/ggml.c.i
.PHONY : ggml.c.i

ggml.s: ggml.c.s
.PHONY : ggml.s

# target to generate assembly for a file
ggml.c.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/ggml.c.s
.PHONY : ggml.c.s

llama.o: llama.cpp.o
.PHONY : llama.o

# target to build an object file
llama.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama.cpp.o
.PHONY : llama.cpp.o

llama.i: llama.cpp.i
.PHONY : llama.i

# target to preprocess a source file
llama.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama.cpp.i
.PHONY : llama.cpp.i

llama.s: llama.cpp.s
.PHONY : llama.s

# target to generate assembly for a file
llama.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama.cpp.s
.PHONY : llama.cpp.s

llama_context.o: llama_context.cpp.o
.PHONY : llama_context.o

# target to build an object file
llama_context.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_context.cpp.o
.PHONY : llama_context.cpp.o

llama_context.i: llama_context.cpp.i
.PHONY : llama_context.i

# target to preprocess a source file
llama_context.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_context.cpp.i
.PHONY : llama_context.cpp.i

llama_context.s: llama_context.cpp.s
.PHONY : llama_context.s

# target to generate assembly for a file
llama_context.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_context.cpp.s
.PHONY : llama_context.cpp.s

llama_eval.o: llama_eval.cpp.o
.PHONY : llama_eval.o

# target to build an object file
llama_eval.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_eval.cpp.o
.PHONY : llama_eval.cpp.o

llama_eval.i: llama_eval.cpp.i
.PHONY : llama_eval.i

# target to preprocess a source file
llama_eval.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_eval.cpp.i
.PHONY : llama_eval.cpp.i

llama_eval.s: llama_eval.cpp.s
.PHONY : llama_eval.s

# target to generate assembly for a file
llama_eval.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_eval.cpp.s
.PHONY : llama_eval.cpp.s

llama_load.o: llama_load.cpp.o
.PHONY : llama_load.o

# target to build an object file
llama_load.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_load.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/llama_load.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/llama_load.cpp.o
.PHONY : llama_load.cpp.o

llama_load.i: llama_load.cpp.i
.PHONY : llama_load.i

# target to preprocess a source file
llama_load.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_load.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/llama_load.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/llama_load.cpp.i
.PHONY : llama_load.cpp.i

llama_load.s: llama_load.cpp.s
.PHONY : llama_load.s

# target to generate assembly for a file
llama_load.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_load.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/llama_load.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/ggml.dir/build.make CMakeFiles/ggml.dir/llama_load.cpp.s
.PHONY : llama_load.cpp.s

llama_loader.o: llama_loader.cpp.o
.PHONY : llama_loader.o

# target to build an object file
llama_loader.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_loader.cpp.o
.PHONY : llama_loader.cpp.o

llama_loader.i: llama_loader.cpp.i
.PHONY : llama_loader.i

# target to preprocess a source file
llama_loader.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_loader.cpp.i
.PHONY : llama_loader.cpp.i

llama_loader.s: llama_loader.cpp.s
.PHONY : llama_loader.s

# target to generate assembly for a file
llama_loader.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_loader.cpp.s
.PHONY : llama_loader.cpp.s

llama_memory_mapper.o: llama_memory_mapper.cpp.o
.PHONY : llama_memory_mapper.o

# target to build an object file
llama_memory_mapper.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_memory_mapper.cpp.o
.PHONY : llama_memory_mapper.cpp.o

llama_memory_mapper.i: llama_memory_mapper.cpp.i
.PHONY : llama_memory_mapper.i

# target to preprocess a source file
llama_memory_mapper.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_memory_mapper.cpp.i
.PHONY : llama_memory_mapper.cpp.i

llama_memory_mapper.s: llama_memory_mapper.cpp.s
.PHONY : llama_memory_mapper.s

# target to generate assembly for a file
llama_memory_mapper.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/llama_memory_mapper.cpp.s
.PHONY : llama_memory_mapper.cpp.s

main.o: main.cpp.o
.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i
.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s
.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/main.cpp.s
.PHONY : main.cpp.s

quantize.o: quantize.cpp.o
.PHONY : quantize.o

# target to build an object file
quantize.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/quantize.cpp.o
.PHONY : quantize.cpp.o

quantize.i: quantize.cpp.i
.PHONY : quantize.i

# target to preprocess a source file
quantize.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/quantize.cpp.i
.PHONY : quantize.cpp.i

quantize.s: quantize.cpp.s
.PHONY : quantize.s

# target to generate assembly for a file
quantize.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/quantize.cpp.s
.PHONY : quantize.cpp.s

utils.o: utils.cpp.o
.PHONY : utils.o

# target to build an object file
utils.cpp.o:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/utils.cpp.o
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/utils.cpp.o
.PHONY : utils.cpp.o

utils.i: utils.cpp.i
.PHONY : utils.i

# target to preprocess a source file
utils.cpp.i:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/utils.cpp.i
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/utils.cpp.i
.PHONY : utils.cpp.i

utils.s: utils.cpp.s
.PHONY : utils.s

# target to generate assembly for a file
utils.cpp.s:
	$(MAKE) $(MAKESILENT) -f CMakeFiles/llama.dir/build.make CMakeFiles/llama.dir/utils.cpp.s
	$(MAKE) $(MAKESILENT) -f CMakeFiles/quantize.dir/build.make CMakeFiles/quantize.dir/utils.cpp.s
.PHONY : utils.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... ggml"
	@echo "... llama"
	@echo "... quantize"
	@echo "... ggml.o"
	@echo "... ggml.i"
	@echo "... ggml.s"
	@echo "... llama.o"
	@echo "... llama.i"
	@echo "... llama.s"
	@echo "... llama_context.o"
	@echo "... llama_context.i"
	@echo "... llama_context.s"
	@echo "... llama_eval.o"
	@echo "... llama_eval.i"
	@echo "... llama_eval.s"
	@echo "... llama_load.o"
	@echo "... llama_load.i"
	@echo "... llama_load.s"
	@echo "... llama_loader.o"
	@echo "... llama_loader.i"
	@echo "... llama_loader.s"
	@echo "... llama_memory_mapper.o"
	@echo "... llama_memory_mapper.i"
	@echo "... llama_memory_mapper.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... quantize.o"
	@echo "... quantize.i"
	@echo "... quantize.s"
	@echo "... utils.o"
	@echo "... utils.i"
	@echo "... utils.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

