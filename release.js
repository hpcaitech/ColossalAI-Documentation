#!/usr/bin/env node
'use strict';

const { ArgumentParser } = require('argparse')
const { execSync } = require('child_process')
const fs = require('fs')
const path = require('path')
const sidebar = require('./sidebars')


function updateDocs(locale, version) {
    const docsRoot = path.join('i18n', locale, 'docusaurus-plugin-content-docs')
    fs.cpSync(path.join(docsRoot, 'current'), path.join(docsRoot, `version-${version}`), { recursive: true })
}

function updateSidebar(version) {
    const json = JSON.stringify(sidebar, null, 2)
    fs.writeFileSync(`versioned_sidebars/version-${version}-sidebars.json`, json)
}

function updateSymlink(version) {
    const cwd = process.cwd()
    process.chdir('versioned_docs')
    fs.symlinkSync(`../i18n/en/docusaurus-plugin-content-docs/version-${version}`, `version-${version}`)
    process.chdir(cwd)
}

function updateVersion(version) {
    const versions = JSON.parse(fs.readFileSync('versions.json'))
    versions.unshift(version)
    fs.writeFileSync('versions.json', JSON.stringify(versions, null, 2))
}

function main() {
    const parser = new ArgumentParser({
        description: 'A script to release a new version.'
    })
    parser.add_argument('version', { help: 'The version you want to release. Do start the version with a "v" for consistency. For example, v0.1.0' })
    const args = parser.parse_args()
    updateSidebar(args.version)
    updateDocs('en', args.version)
    updateDocs('zh-Hans', args.version)
    updateSymlink(args.version)
    updateVersion(args.version)
    execSync('yarn run docusaurus write-translations --locale en')
    execSync('yarn run docusaurus write-translations --locale zh-Hans')
}

main()
