// Small script used to calculate the matrix of tests that are going to be
// performed for a CI run.
//
// This is invoked by the `determine` step and is written in JS because I
// couldn't figure out how to write it in bash.

const fs = require('fs');

// Our first argument is a file that is a giant json blob which contains at
// least all the messages for all of the commits that were a part of this PR.
// This is used to test if any commit message includes a string.
const commits = fs.readFileSync(process.argv[2]).toString();

// The second argument is a file that contains the names of all files modified
// for a PR, used for file-based filters.
const names = fs.readFileSync(process.argv[3]).toString();

// This is the full matrix of what we test on CI. This includes a number of
// platforms and a number of cross-compiled targets that are emulated with QEMU.
// This must be kept tightly in sync with the `test` step in `main.yml`.
//
// The supported keys here are:
//
// * `os` - the github-actions name of the runner os
// * `name` - the human-readable name of the job
// * `filter` - a string which if `prtest:$filter` is in the commit messages
//   it'll force running this test suite on PR CI.
// * `target` - used for cross-compiles if present. Effectively Cargo's
//   `--target` option for all its operations.
// * `gcc_package`, `gcc`, `qemu`, `qemu_target` - configuration for building
//   QEMU and installing cross compilers to execute a cross-compiled test suite
//   on CI.
// * `isa` - changes to `cranelift/codegen/src/$isa` will automatically run this
//   test suite.
// * `rust` - the Rust version to install, and if unset this'll be set to
//   `default`
const array = [
  {
    "os": "windows-latest",
    "name": "Test Windows MSVC x86_64",
    "filter": "windows-x64"
  }
];

for (let config of array) {
  if (config.rust === undefined) {
    config.rust = 'default';
  }
}

function myFilter(item) {
  if(item.os==='windows-latest'){
    return true
  } else {
    return false;
  }
  // if (item.isa && names.includes(`cranelift/codegen/src/isa/${item.isa}`)) {
  //   return true;
  // }
  // if (item.filter && commits.includes(`prtest:${item.filter}`)) {
  //   return true;
  // }

  // // If any runtest was modified, re-run the whole test suite as those can
  // // target any backend.
  // if (names.includes(`cranelift/filetests/filetests/runtests`)) {
  //   return true;
  // }

  return false;
}

const filtered = array.filter(myFilter);

// If the optional third argument to this script is `true` then that means all
// tests are being run and no filtering should happen.
if (process.argv[4] == 'true') {
  console.log(JSON.stringify(array));
  return;
}

// If at least one test is being run via our filters then run those tests.
if (filtered.length > 0) {
  console.log(JSON.stringify(filtered));
  return;
}

// Otherwise if nothing else is being run, run the first one which is Ubuntu
// Linux which should be the fastest for now.
console.log(JSON.stringify([array[0]]));
